### Data specifications

The spatial coverage of the datasets provided is described below:

Features: (240, 150) ------- Label: (800, 500)

```complete-spatial-coverage.yml
data_configuration:
  features_configuration:
    spatial_coverage:
      longitude: [ -20.5, 39.25 ]
      latitude: [ 66.25, 29 ]
  label_configuration:
    spatial_coverage:
      longitude: [ -10, 29.95]
      latitude: [ 60, 35.05 ]
```

During the development stage, a subset of the data is used to validate the implementation of the model:

Features: (32, 20)------- Label: (32, 20)

```reduce-spatial-coverage.yml
data_configuration:
  features_configuration:
    spatial_coverage:
      longitude: [ 6.0, 13.75 ]
      latitude: [ 50, 45.25 ]
  label_configuration:
    spatial_coverage:
      longitude: [ 9.2, 10.75]
      latitude: [ 48, 47.05 ]
```
