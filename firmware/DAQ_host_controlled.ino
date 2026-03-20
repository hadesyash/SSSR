#include "Arduino.h"
#include "analog.h" 
#include "FSPTimer.h" 

#define NUM_CHANNELS        4
#define BYTES_PER_SAMPLE    (NUM_CHANNELS * sizeof(uint16_t))

#define HALF_BUFFER_BYTES   256
#define FULL_BUFFER_BYTES   512

#define SAMPLES_PER_HALF    (HALF_BUFFER_BYTES / BYTES_PER_SAMPLE)
#define TOTAL_SAMPLES       (2 * SAMPLES_PER_HALF)

volatile uint16_t data[TOTAL_SAMPLES * NUM_CHANNELS];

volatile uint16_t sample_index = 0;
volatile bool half_ready = false;
volatile bool full_ready = false;

static FspTimer timer;
volatile int my_count = 0;
volatile int my_count_adc = 0;
// Magic Header (0xDEAD)
const uint8_t header[] = {0xDE, 0xAD};
volatile bool recording_enabled = false;

void my_callback(timer_callback_args_t *args){
  if(args->event == TIMER_EVENT_CYCLE_END){
    my_count++;
    analogStartScan();
  }
}

void my_callback_ADC(uint8_t unit)
{
  if(unit != 0) return;
  if(!recording_enabled) return;

  uint16_t* adc_vals = getAnalogValuesArray();

  uint32_t base = sample_index * NUM_CHANNELS;

  data[base + 0] = adc_vals[0];
  data[base + 1] = adc_vals[1];
  data[base + 2] = adc_vals[2];
  data[base + 3] = adc_vals[9];   // fixed index

  sample_index++;

  if(sample_index == SAMPLES_PER_HALF)
  {
    half_ready = true;
  }
  else if(sample_index == TOTAL_SAMPLES)
  {
    full_ready = true;
    sample_index = 0;
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  
  analogReadResolution(12);
  
  uint8_t type;
  int channel = FspTimer::get_available_timer(type);
  
  timer.begin(TIMER_MODE_PERIODIC, type, channel, 2000.0, 0, my_callback);
  timer.open();
  timer.enable_overflow_irq();
  timer.setup_overflow_irq(1);
  timer.start();  
  
  analogAddPinToGroup(A0);
  analogAddPinToGroup(A1);
  analogAddPinToGroup(A2);
  analogAddPinToGroup(A3);
  
  attachScanEndIrq(my_callback_ADC, ADC_MODE_SINGLE_SCAN, 5);
}

void handle_serial_commands()
{
  if(Serial.available())
  {
    char cmd = Serial.read();

    if(cmd == 'S')   // Start
    {
      sample_index = 0;
      half_ready = false;
      full_ready = false;
      recording_enabled = true;
    }
    else if(cmd == 'E')  // End
    {
      uint16_t samples_to_flush;
      recording_enabled = false;      // stop new writes
      samples_to_flush = sample_index; 
      sample_index = 0;
      half_ready = false;
      full_ready = false;
    }
  }
}

void loop() {
    handle_serial_commands();

    if(half_ready) {
        Serial.write(header, 2);  // Send header first
        Serial.write(
            (uint8_t*)data,
            SAMPLES_PER_HALF * NUM_CHANNELS * sizeof(uint16_t)
        );
        half_ready = false;
    }

    if(full_ready) {
        Serial.write(header, 2);  // Send header first
        Serial.write(
            (uint8_t*)(data + SAMPLES_PER_HALF * NUM_CHANNELS),
            SAMPLES_PER_HALF * NUM_CHANNELS * sizeof(uint16_t)
        );
        full_ready = false;
    }
}