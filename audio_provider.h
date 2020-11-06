#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_AUDIO_PROVIDER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_AUDIO_PROVIDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

//The reference implementation can have no platform dependencies, so it just
//returns an array filled with zeros. For real appications, it needs to ensure
//there'a specialized implementation that accesses hardware APIs.
TfLiteStatus GetAudioSamples(
  tflite::ErrorReporter* error_reporter,
  int start_ms, int duration_ms,
  int* audio_samples_size, int16_t** audio_samples
);

int32_t LatestAudioTimestamp();

#endif
