#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif

#ifndef ARDUINO_EXCLUDE_CODE

#include "tensorflow/lite/micro/examples/Wake_Word/micro_features/micro_model_settings.h"

namespace {
  bool g_is_audio_initialized = false;
  //An internal buffer able to fit 16x sample size.
  constexpr int kAudioCaptureBufferSize = DEFAULT_PDM_BUFFER_SIZE * 16;
  int16_t g_audio_capture_buffer[kAudioCaptureBufferSize];
  //A buffer that holds output.
  int16_t g_audio_output_buffer[kMaxAudioSampleSize];
  //Mark as volatile so we can check in a while loop to see if any samples have
  //arrived yet.
  volatile int32_t g_latest_audio_timestamp = 0;
} //namespace

void CaptureSamples() {
  //This is how many bytes of new data we have each time this is called.
  const int number_of_samples = DEFAULT_PDM_BUFFER_SIZE;
  //Calculate what timestamp the last audio sample represents
  const int32_t time_in_ms =
    g_latest_audio_timestamp +
    (number_of_samples / (kAudioSampleFrequency / 1000));


}
