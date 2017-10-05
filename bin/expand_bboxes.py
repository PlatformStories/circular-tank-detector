import geojson
import numpy as np


def expand_bboxes(input_file, output_file, expand_prop=0.3):
    '''
    Expand the sides of a collection of bounding boxes by a given proportion.

    Args:
        input_file (str) : name of the input geojson file with bounding boxes
        output_file (str) : name under which to save the expanded bounding boxes
        expand_prop (float) : proportion by which to expand the sides of each bounding
            box. Defaults to 0.3.
    '''

    # Open vector file
    with open(input_file) as f:
        bboxes = geojson.load(f)

    expanded_feats = []

    for bbox in bboxes['features']:

        # Get original coordinates
        geom = bbox['geometry']['coordinates'][0]
        maxx, minx = max([coord[0] for coord in geom]), min([coord[0] for coord in geom])
        maxy, miny = max([coord[1] for coord in geom]), min([coord[1] for coord in geom])

        # Calculate new coords
        add_to_x, add_to_y = (expand_prop / 2.) * (maxx - minx), (expand_prop / 2.) * (maxy - miny)
        lx, rx = minx - add_to_x, maxx + add_to_x
        ly, uy = miny - add_to_y, maxy + add_to_y

        bbox['geometry']['coordinates'] = [[[lx, uy], [rx, uy], [rx, ly], [lx, ly], [lx, uy]]]
        expanded_feats.append(bbox)

    # Save expanded bbox feature collection
    bboxes['features'] = expanded_feats
    with open(output_file, 'w') as f:
        geojson.dump(bboxes, f)
