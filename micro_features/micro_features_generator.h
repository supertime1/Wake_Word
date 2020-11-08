#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_FEATURES_GENERATOR_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_FEATURES_GENERATOR_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

//Set up any resources needed for the feature generation pipeline.
TfLiteStatus InitializeMicroFeatures(tflite::ErrorReporter* error_reporter);

//Converts audio sample data into spectrogram.
TfLiteStatus GenerateMicroFeatures(
  tflite::ErrorReporter* error_reporter,
  const int16_t* input, int input_size,
  int output_size, int8_t* output,
  size_t* num_samples_read
);

#endif
