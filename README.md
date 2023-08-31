# PitchNet
Neural network for humain voice pitch analysis.

Code from the paper (PitchNet: A Fully Convolutional Neural Network for Pitch Estimation)[https://arxiv.org/abs/2308.07170]

## Build datasets

The process in itself is quite complex, but its usage is straight forward.

1) Make sure you have a clean repository (no modifications to code, no untracked files)
2) Run ./tools/build-datasets-release.sh

## Build models

You can build a model  y running the pitchnet.ipynb notebook in `./models/` directory.

Alternatively, models can be build from the root directory using `./models/build_models.sh`.

See [the models readme](./models/README.md) for more details.

## Tests
To run tests, just load the test module.
```
python3 -m unittest
```
or
```
python3 -m nose tests
```

To benchmark tests, you can use nose.
```
python3 -m nose tests --with-timer
```

## Documentation
To generate documentation in the docs folder,
run the specific script.
```
./docs/generate.sh
```
