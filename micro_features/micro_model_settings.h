#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_

//512 windows size to pass to FFT, and 30ms of 16kHz inputs (480 samples), by using zero padding to make it 512.
constexpr int kMaxAudioSampleSize = 512;
constexpr int kAudioSampleFrequency = 16000;

//The following values are derived from values used during model training
//update those values if model is changed
constexpr int kFeatureSliceSize = 40;
constexpr int kFeatureSliceCount = 49;
constexpr int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);
constexpr int kFeatureSliceStrideMs = 20;
constexpr int kFeatureSliceDurationMs = 30;

//Variables for the model's output categories.
constexpr int kSilenceIndex = 0;
constexpr int kUnknownIndex = 1;

//If output categories is modified, update the following values.
constexpr int kCategoryCount = 4;
extern const char* kCategoryLabels[kCategoryCount];

#endif
