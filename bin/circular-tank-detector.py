import numpy as np
import geojson, json
import time, os, shutil, re, ast
import protogen
import utm
import cv2
import subprocess

from expand_bboxes import *
from glob import glob
from osgeo import gdal, osr
from keras.models import load_model
from gbdx_task_interface import GbdxTaskInterface
from multiprocessing import Pool, cpu_count
from os.path import join


def preprocess_input(imgs):
    '''
    Preprocess the data by subtracting the mean value from each band
    data_r (4D numpy array): array of images in BGR order (opened with cv2)
    '''
    imgs[:,:,:,0] -= 103.939
    imgs[:,:,:,1] -= 116.779
    imgs[:,:,:,2] -= 123.68
    return imgs


def resize_image(path, side_dim):
    '''
    Open and resize images to input shape. If images are not square, they will
        be warped to the appropriate input shape

    args
    -----
    paths (list): list of paths to input images
    side_dim (int): size (px) of the side dimensions to reshape to
    '''
    img = cv2.imread(path)
    resized = cv2.resize(img, (side_dim, side_dim))
    return resized.astype('float32')


def get_utm_info(image):
    'Return UTM number and proj4 format of utm projection. Image must be in UTM projection.'
    sample = gdal.Open(image)
    prj = sample.GetProjectionRef()
    srs = osr.SpatialReference(wkt=prj)
    return srs.GetUTMZone(), srs.ExportToProj4()

def execute_this(command):
    proc = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.communicate()


class TankDetector(GbdxTaskInterface):
    'Deploys a trained CNN classifier on protogen-generated candidate regions to determine which ones contain tanks.'

    def __init__(self):

        GbdxTaskInterface.__init__(self)

        # Image inputs
        self.ps_dir = self.get_input_data_port('ps_image')

        # Point to imgs. If there are multiple tif's in multiple subdirectories, pick one.
        self.ps_image_path, self.ps_image = [(dp, f) for dp, dn, fn in os.walk(self.ps_dir) for f in fn if 'tif' in f][0]

        # Point to model file if it's there. If not, use locally stored model.
        try:
            self.model = glob(os.path.join(self.get_input_data_port('model'), '*'))[0]
        except:
            self.model = '/model.h5'

        # String inputs
        self.threshold = float(self.get_input_string_port('threshold', '0.5'))
        self.min_compactness = float(self.get_input_string_port('min_compactness', '0.65'))
        self.min_size = int(self.get_input_string_port('min_size', '100'))
        self.max_size = int(self.get_input_string_port('max_size', '12000'))
        self.ptaug = ast.literal_eval(self.get_input_string_port('prediction_time_aug', 'False'))

        # Create output directories
        self.detections_dir = self.get_output_data_port('detections')
        if not os.path.exists(self.detections_dir):
            os.makedirs(self.detections_dir)
        self.candidates_dir = self.get_output_data_port('candidates')
        if not os.path.exists(self.candidates_dir):
            os.makedirs(self.candidates_dir)


    def extract_candidates(self):
        '''
        Use tank extraction protocol to get candidate tanks
        '''
        os.chdir(self.ps_dir)
        imsize = os.path.getsize(self.ps_image)/1e9 # image size in gb

        # convert image to panchromatic
        p = protogen.Interface('radex_scalar','band_fusion')
        p.radex_scalar.band_fusion.operator = 'max'
        p.radex_scalar.band_fusion.threshold = 1.0
        p.image = self.ps_image
        p.image_config.bands = [1,2,3] 
        p.verbose = True
        p.execute()

        panc_out = p.output

        # Get dark candidates
        e = protogen.Interface('extract', 'objects')
        e.extract.objects.type = 'tanks'
        e.extract.objects.shade = 'dark'
        e.extract.objects.visualization = 'binary'
        e.extract.objects.multiplier = 5.0
        e.extract.objects.dark_hole_size = 20
        e.extract.objects.dark_line_radius = 2.0
        e.extract.objects.bright_line_radius = 1.0
        e.extract.objects.bright_patch_size = 20

        e.athos.tree_type = 'max_tree'
        e.athos.area.usage = ['remove if outside']
        e.athos.area.min = [self.min_size]
        e.athos.area.max = [self.max_size]
        e.athos.compactness.usage = ['remove if less']
        e.athos.compactness.min = [self.min_compactness]
        e.athos.compactness.export = [1]

        if imsize > 0.5: # Adjust num tiles based on image size
            e.image_config.number_of_tiles = np.ceil(imsize * 2)
            e.image_config.mosaic_method = 'max'
        e.verbose = True
        e.image = panc_out
        e.execute()

        dark_out = re.sub(str(int(np.ceil(imsize * 2))), 'MOSAIC', e.output)

        # Get bright candidates
        e.extract.objects.shade = 'bright'
        e.athos.tree_type = 'max_tree'
        e.execute()

        bright_out = re.sub(str(int(np.ceil(imsize * 2))), 'MOSAIC', e.output)

        v = protogen.Interface('vectorizer', 'bounding_box')
        v.vectorizer.bounding_box.filetype = 'geojson'
        v.vectorizer.bounding_box.target_bbox = False
        v.vectorizer.bounding_box.target_centroid = True
        v.vectorizer.bounding_box.processor = True
        v.athos.tree_type = 'union_find'
        v.athos.area.export = [1]
        v.image = bright_out
        v.verbose = True
        v.execute()
        bright_out = v.output

        v.image = dark_out
        v.execute()
        dark_out = v.output


        # combine dark and bright features, assign feature id
        with open(bright_out) as f:
            bright = geojson.load(f)

        with open(dark_out) as f:
            dark = geojson.load(f)

        ct = 0
        bright['features'] += dark['features']
        for i in bright['features']:
            i['properties']['feature_id'] = ct
            ct += 1

        with open(join(self.candidates_dir, 'candidates.geojson'), 'w') as f:
            geojson.dump(bright, f)

        # expand candidate bboxes for context
        expand_bboxes(join(self.candidates_dir, 'candidates.geojson'),
                      '/candidates.geojson', 1.0)
        os.chdir('/')


    def extract_chips(self):
        'Extract chips from pan-sharpened image.'
        cmds = []

        # Get UTM info for conversion
        utm_num, utm_proj4 = get_utm_info(join(self.ps_image_path, self.ps_image))

        with open('/candidates.geojson') as f:
            feature_collection = geojson.load(f)['features']

        # Create directory for storing chips
        chip_dir = join(self.ps_image_path, '/chips/')
        if not os.path.exists(chip_dir):
            os.makedirs(chip_dir)

        for feat in feature_collection:
            # get bounding box of input polygon
            polygon = feat['geometry']['coordinates'][0]
            f_id = feat['properties']['feature_id']
            xs, ys = zip(*polygon)
            ulx, lrx, uly, lry = min(xs), max(xs), max(ys), min(ys)

            # Convert corner coords to UTM
            ulx, uly, utm_num1, utm_let1 = utm.from_latlon(uly, ulx, force_zone_number=utm_num)
            lrx, lry, utm_num2, utm_let2 = utm.from_latlon(lry, lrx, force_zone_number=utm_num)

            # format gdal_translate command
            out_loc = join(chip_dir, str(f_id) + '.jpg')

            cmds.append('gdal_translate -of JPEG -eco -q -projwin {} {} {} {} {} {} '\
                        '--config GDAL_TIFF_INTERNAL_MASK YES -co TILED='\
                        'YES'.format(ulx, uly, lrx, lry,
                                     join(self.ps_image_path, self.ps_image), out_loc))

        return cmds


    def deploy_model(self):
        'Deploy model.'
        model = load_model(self.model)
        tanks = {}
        chips = glob(join('chips', '*.jpg'))

        # Classify chips in batches
        indices = np.arange(0, len(chips), 100)
        no_batches = len(indices)

        for no, index in enumerate(indices):
            batch = chips[index: (index + 100)]
            X = preprocess_input(np.array([resize_image(chip, 150) for chip in batch]))
            fids = [os.path.split(chip)[-1][:-4] for chip in batch]

            # Deploy model on batch
            print('Classifying batch {} of {}'.format(no+1, no_batches))
            t1 = time.time()

            if self.ptaug:
                all_yprob = [model.predict(np.rot90(m=X, k=nr, axes=(1,2))) for nr in range(4)]
                yprob = list(np.mean(all_yprob, axis=0))
            else:
                yprob = list(model.predict_on_batch(X))

            # create dict of tank fids and certainties
            for ix, pred in enumerate(yprob):
                if pred[0] > self.threshold:
                    tanks[fids[ix]] = pred[0]

            t2 = time.time()
            print 'Batch classification time: {}s'.format(t2-t1)

        # Save results to geojson
        with open('/candidates.geojson') as f:
            data = geojson.load(f)

        # Save all tanks to output geojson
        tank_feats = []
        for feat in data['features']:
            try:
                res = tanks[str(feat['properties']['feature_id'])]
                feat['properties']['tank_certainty'] = np.round(res, 10).astype(float)
                tank_feats.append(feat)
            except (KeyError):
                continue

        data['features'] = tank_feats

        with open('/detections.geojson', 'wb') as f:
            geojson.dump(data, f)

        # return bboxes to original size
        expand_bboxes('/detections.geojson',
                      join(self.detections_dir, 'detections.geojson'), -0.5)


    def invoke(self):

        # Run protogen to get candidate bounding boxes
        print 'Detecting candidates...'
        candidates = self.extract_candidates()

        # Format vector file and chip from pan image
        print 'Chipping...'
        cmds = self.extract_chips()
        p = Pool(cpu_count())
        p.map(execute_this, cmds)
        p.close()
        p.join()

        # Deploy model
        print 'Deploying model...'
        self.deploy_model()


if __name__ == '__main__':
    with TankDetector() as task:
        task.invoke()

