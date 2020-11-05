#include "Wake_Word/main_functions.h"

#include "Wake_Word/audio_provider.h"
#include "Wake_Word/command_responder.h"
#include "Wake_Word/feature_provider.h"
#include "Wake_Word/micro_features/micro_model_settings.h"
#include "Wake_Word/micro_features/model.h"
#include "Wake_Word/recognize_commands.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {
  //names from tf lite
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  //names from functions
  FeatureProvider* feature_provider = nullptr;
  RecognizeCommands* recognizer = nullptr;
  int32_t previous_time = 0;

  //memory
  constexpr int kTensorArenaSize = 10 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  int8_t feature_buffer[kFeatureElementCount];
  int8_t* model_input_buffer = nullptr;
}

void setup(){
  //Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  //Map the model into a usable data structure.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(
      error_reporter,
      "Model provided is schema version %d not equal to supported version %d.",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  //Pull in ops only need by the model.
  //An alternative is to use AllOpsResolver, but it takes more code space;
  //tflite::AllOpsResolver resolver;
  static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  //Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  //Allocate memory from the tensor_arena for the model's tensor
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  //Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1)) ||
  (model_input->dims->data[1] != (kFeatureSliceCount * kFeatureSliceSize)) || (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = model_input->data.int8;

  //Preare to access the audio spectrograms as inputs to the neural network
  static FeatureProvider static_feature_provider(kFeatureElementCount,feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
}

void loop(){
  //Fetch the spectrogram for the current time.

}
