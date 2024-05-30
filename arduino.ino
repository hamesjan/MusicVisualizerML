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

double vReal[SAMPLES];
double vImag[SAMPLES];


// MACHINE LEARNING 

const float fc1_weight[32][11] = {
  {-0.13866796, -0.3240597, -0.32169506, -0.17956969, -0.11081129, -0.10821672, 0.03615924, -0.00379317, 0.2670995, -0.02124299, 0.28421867},
{0.3855531, -0.34803987, -0.18897267, -0.01362448, -0.24270439, -0.10926255, -0.06513305, -0.01634002, -0.11783354, 0.02444209, 0.13911064},
{0.20056024, -0.3745603, 0.1539765, -0.27724707, -0.27359217, -0.15370497, 0.00166143, -0.22613981, -0.14394821, -0.30065545, 0.09290977},
{-0.21881127, 0.1274659, 0.18452723, -0.22195402, 0.197908, -0.1394863, 0.02811681, -0.04201538, -0.36201, -0.21736063, -0.30168098},
{0.2938427, 0.3934107, 0.0798445, -0.09564899, 0.4175526, 0.44300607, 0.3547112, 0.19367743, 0.19496769, -0.15346971, 0.38143167},
{0.00857943, -0.15203021, -0.14349544, -0.22568971, 0.14987035, -0.00887999, -0.29287705, 0.21533433, 0.12453057, -0.16067827, -0.19026914},
{-0.02356158, -0.03329698, -0.01454644, -0.3162579, -0.18614629, 0.12526543, 0.1907723, 0.16639851, 0.37395412, -0.32157305, -0.26439208},
{-0.08866658, 0.24203515, 0.2629994, 0.11583684, 0.34682658, 0.312758, -0.07638937, 0.02438807, 0.37630787, -0.25127444, 0.27312467},
{-0.1289056, 0.33478042, 0.24730307, 0.1190567, 0.18293087, -0.12017128, 0.03664896, -0.15913515, -0.06939729, 0.28483078, -0.2505276},
{-0.02650577, 0.12969711, 0.07678889, -0.13346143, -0.17057595, -0.25180802, -0.31702027, -0.16632752, 0.15248325, -0.06754831, -0.06236967},
{0.07153589, -0.01366732, -0.4332314, -0.24839413, -0.29640734, 0.03077585, 0.05051033, -0.3865181, -0.14272146, 0.11874873, -0.1367749},
{0.08317108, 0.31339002, -0.09408011, -0.01540927, -0.21065235, 0.20194381, -0.256998, -0.17765835, -0.10133665, 0.15863937, -0.22414514},
{0.09891468, 0.11029327, -0.18585676, 0.15729457, 0.07650778, 0.18601088, -0.05613972, 0.01004062, -0.27157155, 0.13854264, -0.16502129},
{0.02509156, -0.15067437, 0.12440507, 0.17777458, -0.0322673, -0.16884474, -0.08056857, 0.29147503, -0.0393999, -0.15653393, -0.06041821},
{-0.17056404, 0.19399755, 0.04678142, -0.11106889, 0.00809596, -0.05079228, 0.22056873, 0.10492983, 0.18405508, 0.08910231, -0.0463078},
{0.25270873, 0.38866895, 0.16659012, 0.3906433, 0.38639858, 0.24398789, -0.08898899, 0.05936345, 0.06435545, 0.13207822, 0.05074485},
{-0.2968258, 0.2044076, 0.3377397, 0.3978077, 0.03613463, 0.16663615, -0.04372996, 0.10558484, 0.29745546, -0.10082912, -0.02766231},
{-0.3364208, -0.18326767, -0.03764882, 0.15504828, -0.0670422, 0.3590519, 0.30151984, 0.13729243, -0.08153002, -0.02885684, -0.14210857},
{0.16557482, 0.07472656, -0.1359252, 0.25090083, -0.03230617, -0.0887491, -0.16992846, 0.01121654, -0.0025575, 0.10962666, 0.21812077},
{0.19496024, -0.13521874, -0.19401738, 0.03933948, -0.19964543, 0.02175793, -0.05556716, -0.2255406, 0.00422616, 0.20769027, 0.0736094},
{-0.21642864, 0.3406018, 0.29251277, 0.12280519, -0.22981599, 0.07268202, -0.07504209, -0.21906297, -0.14069615, -0.22308339, 0.00199896},
{-0.0372477, 0.2540336, 0.0637835, -0.3068645, -0.14768213, -0.08085307, 0.09107522, 0.09631263, 0.0073819, 0.19024625, -0.14454857},
{0.27182585, -0.1553111, -0.36173695, -0.22623076, -0.00803977, 0.07624128, -0.40508056, -0.08903708, 0.04638169, -0.1420455, 0.34264314},
{-0.2894688, -0.06746291, -0.23726797, 0.09875605, -0.051418, -0.03216118, -0.16890474, 0.1183987, -0.03696302, 0.15775095, -0.2612244},
{-0.26292518, 0.24814796, 0.2630207, 0.27338967, 0.16717792, -0.06335146, 0.08969508, -0.26702237, 0.14523667, -0.03880801, -0.26049873},
{0.11900643, 0.4201387, 0.37981823, 0.25459552, -0.02064182, 0.32496092, 0.3549498, -0.1119941, 0.06117084, -0.1439283, -0.28059366},
{-0.30398163, 0.10325144, 0.12368674, -0.03342127, 0.06023208, -0.24057217, 0.3131527, 0.30683675, -0.01504896, -0.20860311, -0.13681489},
{-0.09279145, -0.13964488, 0.32563066, -0.0485317, 0.23461907, -0.2891503, 0.04917866, -0.15580745, -0.1324674, -0.16611049, -0.06472405},
{-0.30609182, 0.05133162, -0.00814434, 0.00398895, 0.11679252, 0.10128943, -0.24115875, 0.04495031, 0.29194835, -0.12679686, 0.04711676},
{0.35597908, 0.22407559, 0.3998408, 0.2838364, 0.03784354, 0.20803922, -0.04004972, 0.01644284, -0.01317625, -0.09310602, 0.32197762},
{0.25301528, 0.36485767, 0.33675814, 0.23593354, 0.44787964, 0.08302454, 0.2228068, 0.30599818, 0.24724972, 0.18129843, 0.24528818},
{0.12718691, -0.18870874, -0.35905528, -0.3983374, -0.3369612, 0.18133838, 0.22130269, 0.0114398, 0.2176222, -0.18421759, -0.15751281}
};
 

const float fc1_bias[32] = {0.11376391, 0.00444212, -0.09504519, 0.03197620, -0.06794665, -0.13932517, -0.04885804, 0.06664886, 0.17930090, 0.11083263, -0.00913646, -0.21185860, -0.27944592, -0.02442654, -0.10177936, 0.40720570, 0.20019843, 0.15629450, -0.18533307, 0.05099045, 0.16914450, -0.32210400, 0.01060727, 0.21457626, -0.20965092, 0.31835200, 0.38602403, 0.22480705, 0.08395468, 0.07847536, 0.45073690, 0.08783364};

const float fc2_weight[3][32] = {
  { 0.10317414,  0.26343364,  0.25543073,  0.24627759,  0.14257434, -0.01307223,
   0.1383488,   0.27103463,  0.3126706,   0.09530362,  0.20360592, 0.03112742,
  -0.03642166, -0.06214008,  0.04692936,  0.3066122,   0.30271593,  0.08722947,
  -0.00299817,  0.03420687,  0.27509388, -0.09979207,  0.2446814,   0.07026297,
   0.03423055,  0.28321898,  0.27255344,  0.29604945,  0.03936701,  0.19208436,
   0.17920133,  0.1851221 },
 { 0.1614461,   0.17808008,  0.21075068,  0.00859709,  0.263957,    0.25418082,
  -0.00985087,  0.24967691, -0.08352612,  0.2631581,   0.0843875,  -0.00666669,
  -0.04325892, -0.01456889, -0.01211676,  0.10776336, -0.02386189, -0.06709047,
   0.06466015,  0.21427606,  0.18575634, -0.07723746,  0.23007879,  0.25418192,
  -0.07900388,  0.07559273, -0.04006867,  0.03235977, -0.00816943,  0.0683405,
  -0.05547978,  0.12143651},
 { 0.10302731,  0.2134159,   0.0435452,  0.00098088,  0.01532726, -0.05530745,
   0.03186487,  0.09284995, -0.02128142,  0.11664698,  0.29041597,  0.08991107,
   0.11849014,  0.22555737,  0.06813805,  0.2025638,   0.05041533,  0.18298525,
  -0.03308547,  0.02872403,  0.29225805,  0.20001577,  0.15079343,  0.19309808,
   0.2621936,   0.04486175, -0.01102988,  0.00108009,  0.24982236,  0.22522888,
   0.05612918,  0.10610018}
};

const float fc2_bias[3] = {0.1215928,  -0.00698705,  0.1432952};
float rgb[3] = {0, 0, 0};

// ReLU activation function
float maxFloat(float a, float b) {
    return (a > b) ? a : b;
}

float relu(float x) {
    return maxFloat(0.0f, x);
}

// Function to perform inference
void infer_rgb(float currFreq, float freqs[10], float rgb[3]) {
    float x[11];
    x[0] = currFreq * 1.5;
    for (int i = 0; i < 10; i++) {
        x[i + 1] = freqs[i];
    }

    // First layer forward pass
    float h1[32];
    for (int i = 0; i < 32; i++) {
        h1[i] = fc1_bias[i];
        for (int j = 0; j < 11; j++) {
            h1[i] += fc1_weight[i][j] * x[j];
        }
        h1[i] = relu(h1[i]);
    }

    // Second layer forward pass
    for (int i = 0; i < 3; i++) {
        rgb[i] = fc2_bias[i];
        for (int j = 0; j < 32; j++) {
            rgb[i] += fc2_weight[i][j] * h1[j];
        }
    }
}

void copyQueueToArray(cppQueue &queue, float freqs[]) {
    for (int i = 0; i < 10; i++) {
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

  // unsigned long currentMillis = millis();
    
  //   if (currentMillis - previousMillis >= interval) {
  //     previousMillis = currentMillis;
  //     USBSerial.print("CurrFrequency: ");
  //     USBSerial.print(currFreq);
  //     USBSerial.print(" All Frequencies in Queue: "); //FIFO
  //     int queueSize = freqQueue.getCount();
  //     for (int i = 0; i < queueSize; i++) {
  //       float freq;
  //       freqQueue.pop(&freq); // Pop the element from the queue
  //       USBSerial.print(freq, 2); //last one in, last out out, read it from right to left 
  //       USBSerial.print(", ");
  //       freqQueue.push(&freq); // Push the element back to the queue
  // }
  //   USBSerial.println();

  //   }
  

 if (!freqQueue.isEmpty()) {  
    float freq1;
    freqQueue.peek(&freq1);
    int hue = map(freq1, 0, SAMPLES / 2, 0, 255); // Map frequency to hue (0-255)

    
    float freqs[10];
    copyQueueToArray(freqQueue, freqs);

    infer_rgb(currFreq, freqs, rgb);
    USBSerial.print("r: ");
    USBSerial.print(rgb[0], 2); //last one in, last out out, read it from right to left 
    USBSerial.print(" g: ");
    USBSerial.print(rgb[1], 2); //last one in, last out out, read it from right to left 
    USBSerial.print(" b: ");
    USBSerial.print(rgb[2], 2); //last one in, last out out, read it from right to left 
    USBSerial.println();
    color = matrix.colorHSV((int)rgb[0], (int)rgb[1], (int)rgb[2]); // Convert hue to RGB color

    // Draw the spectrogram on the matrix with the calculated color
    drawSpectrogram(color);
  }
}
