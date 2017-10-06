# storage-tank-detector

A GBDX task that detects tanks. Tanks are circular structures for storing oil, water or gas.

The input to the task is pan-sharpened image in UTM projection. The output is a geojson file with the detection bounding boxes.



## Run

This is a sample workflow to detect tanks in the United Arab Emirates. The required input imagery is found in S3.

1. Within an iPython terminal create a GBDX interface an specify the task input location:

    ```python
    from gbdxtools import Interface
    from os.path import join
    import uuid

    gbdx = Interface()

    input_location = 's3://gbd-customer-data/32cbab7a-4307-40c8-bb31-e2de32f940c2/platform-stories/storage-tank-detector/'
    ```

2. Create a task instance and set the required [inputs](#inputs):

    ```python
    td = gbdx.Task('storage-tank-detector')
    td.inputs.ps_image = join(input_location, 'ps-image')
    td.inputs.min_size = '50'
    ```

3. Create a workflow instance and specify where to save the output:

    ```python
    wf = gbdx.Workflow([td])
    random_str = str(uuid.uuid4())
    output_location = join('platform-stories/trial-runs', random_str)

    wf.savedata(td.outputs.detections, join(output_location, 'tank_detections'))
    wf.savedata(td.outputs.candidates, join(output_location, 'tank_candidates'))
    ```

5. Execute the workflow:

    ```python
    wf.execute()
    ```

6. Track the status of the workflow as follows:

    ```python
    wf.status
    ```

## Algorithm

The task does the following:

+ Computes a max-tree structure of the input image.
+ Filters based on defined size and shape constraints that are characteristic of tanks, to produce candidate bounding boxes.
+ Chips out the candidates from the pan-sharpened image and feeds them to a Keras model, which classifies each candidate as 'Tank' or 'Other'. If a model is not provided as input, the task uses a default model built into the container.



## Inputs

GBDX input ports can only be of "directory" or "string" type. Booleans, integers and floats are passed to the task as strings, e.g., "True", "10", "0.001".

| Name  | Type | Description | Required |
|---|---|---|---|
| ps_image | directory | Contains a 3-band pan-sharpened image in geotiff format and UTM projection. This directory should contain only one image otherwise one is selected arbitrarily. | True |
| model | directory | Contains a keras model in h5 format. | False |
| threshold | string | Decision threshold. Defaults to 0.5. | False |
| prediction_time_aug | string | When deploying the model rotate each chip 4 times and average all predictions. This will increase deployment time by a factor of four, but tends to result in higher accuracy. Defaults to False. | False |
| min_compactness | string | Minimum compactness of a feature to qualify as a tank candidate. Default is 0.65. | False |
| min_size | string | The minimum area in m2 to qualify as a tank candidate. Default is 100. | False |
| max_size | string | Maximum area in m2 to qualify as a tank candidate. Default is 12000. | False |


## Outputs

| Name  | Type | Description |
|---|---|---|
| detections | directory | Contains geojson file with detection bounding boxes. |
| candidates | directory | Contains geojson file with candidate bounding boxes. |

## Comments/Recommendations

+ If precision is more important than recall then increase the threshold, and vice versa.
+ Increading the size or compactness range will increase the run time, as more candidates are retrieved.
+ The required projection for the input images is UTM, due to the fact that candidate locations are derived based on geometrical properties such as size and compactness.
+ Metallic tanks with a sun reflection or tanks with a color gradient may be missed by the max-tree.
+ The compactness parameter represents how close an object is to a perfect circle (as an ideal tank should be). However, rust, shadows, and surrounding objects may contribute to a lower compactness. Allow some leeway when defining the minimum acceptable compactness to account for these inconsistencies.

## Changelog

### Current version

#### Training

Training data was obtained from 20 WV03 and WV02 imagescolected between 2014 and 2017 over large oil fields. Locations with a diverse array of climates were used: Aden Yemen, Alexandria Egypt, Bahrain, Coatzacoalos Mexico, Santa Elena Ecuador, Gothenburg Sweden, Houston Texas, Kuwait, Liaoning China, Nederland Texas, Osaka Japan, Paraiso Mexico, Rotterdam Netherlands, Suez Egypt, Veracruz Mexico, and Zhenhai China. Chips were extracted from each image using PROTOGEN to get tank and no tank examples. 5000 chips (3:1 no tank:tank) were then manually curated to ensure accuracy and used as training data. The imagery was atmospherically compensated and pan-sharpened. The architecture of the neural network is VGG-16 and was pretrained on the imagenet dataset.

#### Runtime



## Development

### Build the Docker Image

You need to install [Docker](https://docs.docker.com/engine/installation).

Clone the repository:

```bash
git clone https://github.com/platformstories/storage-tank-detector
```

Then build the image locally. Building requires input environment variables for protogen and GBDX AWS credentials. You will need to contact kostas.stamatiou@digitalglobe.com for access to Protogen.

```bash
cd storage-tank-detector
docker build --build-arg PROTOUSER=<GitHub username> \
    --build-arg PROTOPASSWORD=<GitHub password> \
    --build-arg AWS_ACCESS_KEY_ID=<AWS access key> \
    --build-arg AWS_SECRET_ACCESS_KEY=<AWS secret key> \
    --build-arg AWS_SESSION_TOKEN=<AWS session token> \
    -t storage-tank-detector .
```

### Try out locally

You need a GPU and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Create a container in interactive mode and mount the sample input under `/mnt/work/input/`:

```bash
nvidia-docker run -v full/path/to/sample-input:/mnt/work/input -it storage-tank-detector
```

Then, within the container:

```bash
python /storage-tank-detector.py
```

Confirm that the output geojsons are under `/mnt/work/output/`.

### Docker Hub

Login to Docker Hub:

```bash
docker login
```

Tag your image using your username and push it to DockerHub:

```bash
docker tag storage-tank-detector yourusername/storage-tank-detector
docker push yourusername/storage-tank-detector
```

The image name should be the same as the image name under containerDescriptors in storage-tank-detector.json.

Alternatively, you can link this repository to a [Docker automated build](https://docs.docker.com/docker-hub/builds/). Every time you push a change to the repository, the Docker image gets automatically updated.

### Register on GBDX

In a Python terminal:
```python
from gbdxtools import Interface
gbdx = Interface()
gbdx.task_registry.register(json_filename = 'storage-tank-detector.json')
```

Note: If you change the task image, you need to reregister the task with a higher version number in order for the new image to take effect. Keep this in mind especially if you use Docker automated build.
