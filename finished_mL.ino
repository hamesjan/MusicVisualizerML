#include <cppQueue.h>
#include <Adafruit_Protomatter.h>
#include <arduinoFFT.h>


#define SAMPLES 512              // Must be a power of 2
#define SAMPLING_FREQUENCY 44100 // Hz, must be less than 1MHz

#define MIC_PIN 1               // Analog input pin for the microphone

// Define the pin numbers for the matrix connections
#define R1_PIN 46
#define G1_PIN 45
#define B1_PIN 42
#define R2_PIN 41
#define G2_PIN 40
#define B2_PIN 39
#define A_PIN 38
#define B_PIN 37
#define C_PIN 36
#define D_PIN 35
#define E_PIN 48
#define LAT_PIN 34
#define OE_PIN 33
#define CLK_PIN 47

// Initialize arrays for the RGB pins and address pins
uint8_t rgbPins[] = {R1_PIN, G1_PIN, B1_PIN, R2_PIN, G2_PIN, B2_PIN};
uint8_t addrPins[] = {A_PIN, B_PIN, C_PIN, D_PIN, E_PIN};
uint8_t clockPin = CLK_PIN;
uint8_t latchPin = LAT_PIN;
uint8_t oePin = OE_PIN;

// Create an instance of the Protomatter matrix for a 64x64 matrix
Adafruit_Protomatter matrix(
  64,          // Width of matrix (or matrix chain) in pixels
  6,           // Bit depth, 1-6
  1, rgbPins,  // Number of matrix chains, array of 6 RGB pins for each
  5, addrPins, // Number of address pins (height is inferred), array of pins
  clockPin, latchPin, oePin, // Other matrix control pins
  false        // No double-buffering here (see "doublebuffer" example)
);

// Instantiate the FFT object
ArduinoFFT<double> FFT = ArduinoFFT<double>();

unsigned int samplingPeriodUs;
unsigned long microseconds;
unsigned long previousMillis = 0;
const long interval = 1000; 
float currFreq = 0;


uint16_t color;
cppQueue freqQueue(sizeof(float), 10, FIFO); // FIFO: First In First Out
float rgb[3] = {0, 0, 0};

double vReal[SAMPLES];
double vImag[SAMPLES];





// MACHINE LEARNING 

const int INPUT_SIZE = 11;
const int HIDDEN_SIZE = 16;
const int OUTPUT_SIZE = 3;

float fc1_weight[16][11] = {
    {-0.01214661, 0.14070152, -0.12459522, 0.06993938, -0.1081764, -0.19592142, -0.04309722, 0.10967974, -0.222808, -0.05954723, -0.03580753},
    {-0.01281069, 0.05839092, 0.17956163, -0.0041439, -0.2804335, 0.17655624, -0.00711687, -0.22623353, 0.22005, -0.2286026, -0.16122212},
    {0.25610834, 0.5032039, 0.17155366, 0.08113076, 0.10510236, 0.22637652, 0.19963628, -0.14565095, -0.17246167, 0.16333683, 0.1626751},
    {-0.17664498, -0.06944207, 0.32330123, -0.0108783, -0.15040389, -0.16421005, -0.18145092, 0.20788215, -0.1639205, 0.18394712, 0.07955801},
    {-0.05247873, -0.1571798, -0.2812541, -0.30699277, -0.23886609, -0.18911752, -0.14181224, -0.04441196, -0.32245508, -0.1359455, 0.31269673},
    {-0.23534945, 0.01756407, -0.02887606, -0.04792156, -0.19934602, -0.06773981, 0.16493538, 0.10899726, 0.33547148, -0.06313752, -0.00561168},
    {-0.19805871, 0.1533537, -0.14589065, 0.15233964, 0.0543845, -0.3558022, 0.10463741, -0.34611568, -0.03388342, -0.10589151, 0.01832983},
    {0.3953708, -0.03983451, 0.08425223, -0.38784352, -0.25895604, -0.13806482, 0.18998314, -0.21697454, 0.14437184, -0.1053055, -0.21369277},
    {0.18615104, 0.27862215, 0.26610774, 0.08662628, 0.40428385, -0.11153787, -0.18584232, -0.10441718, -0.09153924, -0.20458631, 0.14950402},
    {-0.03315066, 0.1777767, 0.379214, 0.17097367, 0.1439998, 0.4036421, 0.21816605, 0.20759629, 0.08400941, 0.0177246, -0.16849025},
    {-0.17204045, -0.08042252, -0.35076094, 0.11140334, -0.18290651, -0.14955893, 0.09150577, 0.25076526, 0.0803858, 0.304225, 0.11690042},
    {0.17784826, 0.14642772, -0.23842242, 0.19078362, 0.06110914, 0.05353062, -0.09650513, -0.1849847, 0.03956837, 0.1109955, -0.13309121},
    {-0.11721827, -0.31410488, -0.09755631, -0.21191458, 0.4605746, 0.45355004, 0.2951988, 0.06032131, 0.33766845, 0.09452144, 0.25423834},
    {-0.31734347, -0.30929592, 0.00340971, -0.01732022, 0.30196312, 0.4893048, 0.11113961, 0.24400395, -0.21726578, -0.08343723, -0.12443008},
    {-0.08295429, 0.18063284, -0.18162693, 0.12587588, -0.14672942, 0.15641001, -0.20650394, 0.09488977, 0.2687336, 0.5600277, 0.2583301},
    {-0.23339371, -0.22229664, -0.07198399, -0.1479324, 0.19355498, -0.0185208, 0.04732901, -0.1965689, -0.1664933, -0.07563716, 0.31308198}
};

float fc1_bias[16] = {-0.12781498, -0.24727076, 0.18702833, -0.305816, 0.27822503, 0.1702911, -0.01722901, 0.07705813, -0.03841614, 0.17698894, -0.28357106, -0.39097214, 0.06091013, 0.00557305, 0.24485835, -0.2899324};

float fc2_weight[3][16] = {
    {-0.09086456, -0.20924006, 0.23792747, -0.07209299, 0.2448962, 0.08262997, -0.15483944, 0.31029218, 0.14325422, 0.32146376, -0.00108195, -0.11875976, 0.21594505, 0.19532412, 0.2326524, -0.03609438},
    {0.0754042, 0.17370747, -0.15414016, 0.05334909, 0.03831212, -0.131325, 0.17092516, 0.15668242, 0.18193954, 0.18313959, 0.3119366, -0.10487936, 0.06588542, 0.00607965, 0.11571737, -0.15809512},
    {-0.17145868, 0.15691023, 0.1998374, -0.1338659, 0.04171347, 0.14032428, 0.08442702, -0.13417763, -0.0964267, -0.1254138, 0.00345254, -0.1816723, -0.00390419, 0.00337188, -0.09056474, -0.02512853}
};

float fc2_bias[3] = {-0.0617934, -0.11026014, 0.10569768};

float mean[] = { 2686.11111111, 1842.66666667, 1930.25925926, 2035.25925926, 2124,
 2224.92592593, 2348.07407407, 2418.7037037,  2524, 2581.22222222, 2653.07407407 };
float scale[] = { 396.52764448, 1166.4768417,  1125.43336814, 1092.84299955, 1043.27135066,
  985.77493209,  933.678703,   843.3866832 ,  738.92970647,  651.15790503,
  495.19483079 };

float relu(float input) {
    return input < 0 ? 0 : input;
}

// Forward pass function
void forward_pass(float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {
    float hidden[HIDDEN_SIZE] = {0};
    
    // First layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += fc1_weight[i][j] * input[j];
        }
        hidden[i] += fc1_bias[i];
        hidden[i] = relu(hidden[i]);
    }

    // Second layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += fc2_weight[i][j] * hidden[j];
        }
        output[i] += fc2_bias[i];
        // Output is already in the required form
    }
}

void normalize_input(float input[INPUT_SIZE]) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = (input[i] - mean[i]) / scale[i];
    }
}

void copyQueueToArray(cppQueue &queue, float freqs[]) {
    freqs[0] = currFreq;
    for (int i = 1; i < 11; i++) {
        float value;
        if (freqQueue.pop(&value)) { // Dequeue the element
            freqs[i] = value; // Copy the element to the array
            freqQueue.push(&value); // Enqueue the element back to the queue
        }
    }
}


void setup() {
  USBSerial.begin(9600); // Initialize serial communication for debugging (USB Serial)
  Serial1.begin(115200, SERIAL_8N1, 17, 18);
  analogReadResolution(12); // Set ADC resolution to 12 bits (0-4095)

  samplingPeriodUs = round(1000000 * (1.0 / SAMPLING_FREQUENCY));

  // Initialize matrix
  matrix.begin();
}

// Function to map FFT results to matrix
void drawSpectrogram(uint16_t color) {
  matrix.fillScreen(0); // Clear the screen first

  randomSeed(analogRead(0));

  // Find the index of the maximum magnitude in the FFT results
  for (int x = 1; x < SAMPLES; x = x + 8) {

    // Map frequency to color and draw on matrix
    int intensity = map(vReal[x], 0, 1000, 0, 16); // Half of the original height (64/2 = 32)
    intensity = constrain(intensity, 0, 16);

    //uint16_t color;
    //int hue = map(x, 0, SAMPLES / 2, 0, 255); // Map frequency to hue (0-255)
    //color = matrix.colorHSV(hue * 256, 255, 255); // Convert hue to RGB color

    int startY = 32; // Middle of the matrix (64/2 = 32)

    for (int y = 0; y < intensity; y++) {
      if ( (x/8) % 2 == 0 ) {
        matrix.drawPixel(x / 8, startY - y, color); // Go up
      } else {
        matrix.drawPixel(x / 8, startY + y, color); // Go down
      }
    }
  }
  matrix.show(); // Update the display
}

void loop() {
  // Sample the microphone signal
  for (int i = 0; i < SAMPLES; i++) {
    microseconds = micros();

    vReal[i] = analogRead(MIC_PIN);
    vImag[i] = 0;

    while (micros() < (microseconds + samplingPeriodUs)) {
      // Wait for the next sample time
    }
  }

  // Compute FFT
  FFT.windowing(vReal, SAMPLES, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(vReal, vImag, SAMPLES, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, SAMPLES);

  if (Serial1.available() >= sizeof(float)) {
    float receivedFreq;
    Serial1.readBytes((byte*)&receivedFreq, sizeof(float));
    // Remove the oldest element if the queue is full
    if (freqQueue.isFull()) {
        float temp;
        freqQueue.pop(&temp);
    }
    // Add the received frequency to the queue
    if (currFreq != 0){
      freqQueue.push(&currFreq);
    }
    currFreq = receivedFreq;
}

  unsigned long currentMillis = millis();
    
    if (currentMillis - previousMillis >= interval) {
      previousMillis = currentMillis;
      USBSerial.print("CurrFrequency: ");
      USBSerial.print(currFreq);
      USBSerial.print(" All Frequencies in Queue: "); //FIFO
      int queueSize = freqQueue.getCount();
      for (int i = 0; i < queueSize; i++) {
        float freq;
        freqQueue.pop(&freq); // Pop the element from the queue
        USBSerial.print(freq, 2); //last one in, last out out, read it from right to left 
        USBSerial.print(", ");
        freqQueue.push(&freq); // Push the element back to the queue
  }
    USBSerial.println();

    }
  

 if (!freqQueue.isEmpty()) {  
    float freqs[11];
    copyQueueToArray(freqQueue, freqs);
    normalize_input(freqs);

    // Output array
    float output[OUTPUT_SIZE] = {0};

    // Perform forward pass
    forward_pass(freqs, output);

    // Convert output to RGB values
    int r = round(output[0] * 255.0);
    int g = round(output[1] * 255.0);
    int b = round(output[2] * 255.0);

    r = min(max(r, 0), 255);
    g = min(max(g, 0), 255);
    b = min(max(b, 0), 255);

    USBSerial.print("r: ");
    USBSerial.print(r); //last one in, last out out, read it from right to left 
    USBSerial.print(" g: ");
    USBSerial.print(g); //last one in, last out out, read it from right to left 
    USBSerial.print(" b: ");
    USBSerial.print(b); //last one in, last out out, read it from right to left 
    USBSerial.println();
    color = matrix.color565(r, g, b); // Convert hue to RGB color

    // Draw the spectrogram on the matrix with the calculated color
    drawSpectrogram(color);
  }
}
