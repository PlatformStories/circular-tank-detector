# circular-tank-detector

A GBDX task that detects circular storage tanks. The input to the task is pan-sharpened image in UTM projection. The output is a geojson file with the detection bounding boxes.

## Run

This is a sample workflow. The required input imagery is found in S3 in the provided location.

1. Within an iPython terminal create a GBDX interface an specify the task input location:

    ```python
    from gbdxtools import Interface
    from os.path import join
    import uuid

    gbdx = Interface()

    input_location = 's3://gbd-customer-data/32cbab7a-4307-40c8-bb31-e2de32f940c2/platform-stories/circular-tank-detector/'
    ```

2. Create a task instance and set the required [inputs](#inputs):

    ```python
    td = gbdx.Task('circular-tank-detector')
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

+ Computes the max-tree from the panchromatic image (which is derived from the input pan-sharpened image).
+ Extracts candidate bounding boxes using the provided minimum compactness value and size range (defaults can also be used).
+ Chips out the candidates from the pan-sharpened image and feeds them to a Keras model, which classifies each candidate as 'tank' or 'other'. If a model is not provided as input, the task uses the default model included in the container.

The model was built with training data was obtained from 20 WV03 and WV02 images collected between 2014 and 2017 over large oil fields. Locations with a diverse array of climates were used: Aden Yemen, Alexandria Egypt, Bahrain, Coatzacoalos Mexico, Santa Elena Ecuador, Gothenburg Sweden, Houston Texas, Kuwait, Liaoning China, Nederland Texas, Osaka Japan, Paraiso Mexico, Rotterdam Netherlands, Suez Egypt, Veracruz Mexico, and Zhenhai China. Positive and negative example chips were automatically extracted from each image using the procedure described [in this blog post](gbdxstories.digitalglobe.com/circular-tanks). 5000 chips (with a 3:1 ratio of positive to negative) were randomly selected and manually curated to correct for false positive and false negative examples. VGG-16, pretrained on ImageNet was trained using these chips. The resulting model is the one included in this task.


## Inputs

GBDX input ports can only be of "directory" or "string" type. Booleans, integers and floats are passed to the task as strings, e.g., "True", "10", "0.001".

| Name  | Type | Description | Required |
|---|---|---|---|
| ps_image | directory | Contains a 3-band pan-sharpened image in geotiff format and UTM projection. This directory should contain only one image otherwise one is selected arbitrarily. | True |
| model | directory | Contains a Keras model in h5 format. | False |
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

+ Keep the input image size smaller than 3GB.
+ If precision is more important than recall then increase the threshold, and vice versa.
+ Increasing the size range and/or decreasing the minimum compactness will increase the run time, as more candidates are retrieved.
+ The required projection for the input images is UTM, due to the fact that candidate locations are derived based on geometrical properties such as size and compactness.
+ False negatives may include tanks with a sun reflection or a color gradient. False positives may include various circular features including ponds and roundabouts.


## Development

### Build the Docker Image

You need to install [Docker](https://docs.docker.com/engine/installation).

Clone the repository:

```bash
git clone https://github.com/platformstories/circular-tank-detector
```

Then build the image locally. 

```bash
cd circular-tank-detector
docker build --build-arg PROTOUSER=<GitHub username> \
    --build-arg PROTOPASSWORD=<GitHub password> \
    --build-arg AWS_ACCESS_KEY_ID=<AWS access key> \
    --build-arg AWS_SECRET_ACCESS_KEY=<AWS secret key> \
    --build-arg AWS_SESSION_TOKEN=<AWS session token> \
    -t circular-tank-detector .
```

### Try out locally

You need a GPU and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Create a container in interactive mode and mount the sample input under `/mnt/work/input/`:

```bash
nvidia-docker run -v full/path/to/sample-input:/mnt/work/input -it circular-tank-detector
```

Then, within the container:

```bash
python /circular-tank-detector.py
```

Confirm that the output geojsons are under `/mnt/work/output/`.

### Docker Hub

Login to Docker Hub:

```bash
docker login
```

Tag your image using your username and push it to DockerHub:

```bash
docker tag circular-tank-detector yourusername/circular-tank-detector
docker push yourusername/circular-tank-detector
```

The image name should be the same as the image name under containerDescriptors in circular-tank-detector.json.

Alternatively, you can link this repository to a [Docker automated build](https://docs.docker.com/docker-hub/builds/). Every time you push a change to the repository, the Docker image gets automatically updated.

### Register on GBDX

In a Python terminal:
```python
from gbdxtools import Interface
gbdx = Interface()
gbdx.task_registry.register(json_filename = 'circular-tank-detector.json')
```

Note: If you change the task image, you need to reregister the task with a higher version number in order for the new image to take effect. Keep this in mind especially if you use Docker automated build.
