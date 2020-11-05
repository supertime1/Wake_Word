#include "Wake_Word/feature_provider.h"

#include "Wake_Word/audio_provider.h"
#include "Wake_Word/micro_features/micro_features_generator.h"
#include "Wake_Word/micro_features/micro_model_settings.h"

FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data):
feature_size_(feature_size),
feature_data_(feature_data),
is_first_run_(true) {
//Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider:: ~FeatureProvider() {}

TfLiteStatus FeatureProvider::PopulateFeatureData(
  tflite::ErrorReporter* error_reporter, int32_t last_time_in_ms, int32_t time_in_ms, int* how_many_new_slices) {
    if (feature_size_ != kFeatureElementCount) {
      TF_LITE_REPORT_ERROR(error_reporter,
        "Requested feature_data_ size %d doesn't match %d",
        feature_size_, kFeatureElementCount);
      return kTfLiteError;
    }

  //Quantize the time into steps as long as each window stride, so we can
  //figure out which audio data we need to fetch.

  }

)
