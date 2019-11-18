#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define DATA_LOCATION "data.csv"
// #define DATA_LOCATION "fakeData.csv"

#define IN_SIZE 2
#define HIDDEN_SIZE 2
#define OUT_SIZE 2

#define EPOCS 1
#define LAMBDA 1
#define ALPHA 0.6
#define ETA 1

#define CAP_DATA false
#define CAP_LIMIT 3000

#define INF 10000000

double r()
{
  double var = (double)rand() / (RAND_MAX);
  double value = (int)(var * 100 + .5);
  return (double)value / 100;
}

double logisticActivation(double val)
{
  return 1.0 / (1 + pow(2.7182, -1 * LAMBDA * val));
}

struct DataSet
{
  vector<vector<double>> data;
  vector<vector<double>> trainData;
  vector<vector<double>> validationData;
  vector<vector<double>> testData;
  double x1Min;
  double x1Max;
  double x2Min;
  double x2Max;
  double y1Min;
  double y1Max;
  double y2Min;
  double y2Max;

  DataSet()
  {
    data = {};
    trainData = {};
    validationData = {};
    testData = {};
    x1Min = INF;
    x1Max = -1 * INF;
    x2Min = INF;
    x2Max = -1 * INF;
    y1Min = INF;
    y1Max = -1 * INF;
    y2Min = INF;
    y2Max = -1 * INF;
  }

  void readData(string fileName)
  {
    string line;
    ifstream file(fileName);
    string x1, x2, y1, y2;
    if (file.is_open())
    {
      while (getline(file, x1, ','))
      {
        getline(file, x2, ',');
        getline(file, y1, ',');
        getline(file, y2, '\n');
        data.push_back({stod(x1), stod(x2), stod(y1), stod(y2)});
        if (stod(x1) > x1Max)
          x1Max = stod(x1);
        if (stod(x1) < x1Min)
          x1Min = stod(x1);
        if (stod(x2) > x2Max)
          x2Max = stod(x2);
        if (stod(x2) < x2Min)
          x2Min = stod(x2);
        if (stod(y1) > y1Max)
          y1Max = stod(y1);
        if (stod(y1) < y1Min)
          y1Min = stod(y1);
        if (stod(y2) > y2Max)
          y2Max = stod(y2);
        if (stod(y2) < y2Min)
          y2Min = stod(y2);
      }
      file.close();
    }
  }
  void shuffle(string subset)
  {
    if (subset == "data")
      random_shuffle(data.begin(), data.end());
    if (subset == "train")
      random_shuffle(trainData.begin(), trainData.end());
    if (subset == "validation")
      random_shuffle(validationData.begin(), validationData.end());
    if (subset == "test")
      random_shuffle(testData.begin(), testData.end());
    return;
  }
  void splitData()
  {
    shuffle("data");
    for (size_t i = 0; i < data.size(); i++)
    {
      if (i < data.size() * 0.1)
        testData.push_back(data[i]);
      else if (i < data.size() * 0.2)
        validationData.push_back(data[i]);
      else
        trainData.push_back(data[i]);
    }
  }
  double normalizeX1(double x1)
  {
    return (x1 - x1Min) / (x1Max - x1Min);
  }
  double normalizeX2(double x2)
  {
    return (x2 - x2Min) / (x2Max - x2Min);
  }
  double normalizeY1(double y1)
  {
    return (y1 - y1Min) / (y1Max - y1Min);
  }
  double normalizeY2(double y2)
  {
    return (y2 - y2Min) / (y2Max - y2Min);
  }
  double denormalizeY1(double y1pred)
  {
    return y1pred * (y1Max - y1Min) + y1Min;
  }
  double denormallizeY2(double y2pred)
  {
    return y2pred * (y2Max - y2Min) + y2Min;
  }
  vector<pair<double, double>> getFeatures(string subset, bool normalized)
  {
    vector<vector<double>> d = data;
    if (subset == "train")
      d = trainData;
    if (subset == "validation")
      d = validationData;
    if (subset == "test")
      d = testData;
    vector<pair<double, double>> features = {};
    for (size_t i = 0; i < d.size(); i++)
    {
      pair<double, double> p = make_pair(d[i][0], d[i][1]);
      if (normalized)
      {
        p.first = this->normalizeX1(p.first);
        p.second = this->normalizeX2(p.second);
      }
      features.push_back(make_pair(p.first, p.second));
    }
    return features;
  }
  vector<pair<double, double>> getLabels(string subset, bool normalized)
  {
    vector<vector<double>> d = data;
    if (subset == "train")
      d = trainData;
    if (subset == "validation")
      d = validationData;
    if (subset == "test")
      d = testData;
    vector<pair<double, double>> labels = {};
    for (size_t i = 0; i < d.size(); i++)
    {
      pair<double, double> p = make_pair(d[i][2], d[i][3]);
      if (normalized)
      {
        p.first = this->normalizeY1(p.first);
        p.second = this->normalizeY2(p.second);
      }
      labels.push_back(make_pair(p.first, p.second));
    }
    return labels;
  }

  void info()
  {
    cout << "data.size(): " << data.size() << endl;
    cout << "trainData.size(): " << trainData.size() << endl;
    cout << "validationData.size(): " << validationData.size() << endl;
    cout << "testData.size(): " << testData.size() << endl;
    cout << "x1Min: " << x1Min << "  x1Max: " << x1Max << endl;
    cout << "x2Min: " << x2Min << "  x2Max: " << x2Max << endl;
    cout << "y1Min: " << y1Min << "  y1Max: " << y1Max << endl;
    cout << "y2Min: " << y2Min << "  y2Max: " << y2Max << endl;
    int count = fmin(5, data.size());
    for (int i = 0; i < count; i++)
    {
      cout << data[i][0] << " " << data[i][1] << " " << data[i][2] << " " << data[i][3] << endl;
    }
  }
};

struct Neuron
{
  int i;
  double inVal;
  double actVal;
  double G;
  double sum;
  vector<double> weights;
  vector<double> deltaWeights;

  Neuron(int numberWeigths)
  {
    inVal = 1;
    actVal = 1;
    G = 1;
    sum = 1;
    vector<double> w = {};
    vector<double> dw = {};
    for (int i = 0; i < numberWeigths; i++)
    {
      w.push_back(r());
      dw.push_back(0);
    }
    weights = w;
    deltaWeights = dw;
  }
};

struct Layer
{
  vector<Neuron> neurons;

  Layer(int layerSize, int numberWeigths, bool bias)
  {
    vector<Neuron> neurons = {};
    for (size_t i = 0; i < layerSize; i++)
    {
      neurons.push_back(Neuron(numberWeigths));
    }
    if (bias && numberWeigths > 0)
    {
      neurons[0].weights[0] = 0;
      neurons[0].weights[1] = 0;
      neurons[0].weights[2] = 0;
    }
    this->neurons = neurons;
  }
};

struct Network
{
  DataSet dataSet;
  vector<Layer> layers;

  Network(int hiddenLayerSize)
  {
    DataSet dataSet = DataSet();
    dataSet.info();
    dataSet.readData(DATA_LOCATION);
    dataSet.shuffle("all");
    dataSet.splitData();
    this->dataSet = dataSet;
    // inputLayer = Layer(3, 0);
    // hiddenLayer = Layer(hiddenLayerSize + 1, 3);
    // Layer outputLayer = Layer(2, hiddenLayerSize + 1);
    vector<Layer> layers = {Layer(3, 0, true),
                            Layer(hiddenLayerSize + 1, 3, true),
                            Layer(2, hiddenLayerSize + 1, false)};
    this->layers = layers;
  }

  void printNetwork()
  {
    cout << "   **********" << endl;
    for (size_t i = 0; i < layers.size(); i++)
    {
      cout << "    Layer " << i << endl;
      for (size_t j = 0; j < layers[i].neurons.size(); j++)
      {
        cout << "inVal: " << layers[i].neurons[j].inVal << "    actVal: " << layers[i].neurons[j].actVal << endl;
        cout << "G: " << layers[i].neurons[j].G << "    sum: " << layers[i].neurons[j].sum << endl;

        cout << "        weigths: ";
        for (size_t k = 0; k < layers[i].neurons[j].weights.size(); k++)
        {
          cout << " " << layers[i].neurons[j].weights[k];
        }
        cout << endl;
      }
    }
    cout << "**********" << endl;
  }

  void forwardProp(pair<double, double> row)
  {
    // Input Layer
    layers[0].neurons[0].inVal = 1;
    layers[0].neurons[1].inVal = row.first;
    layers[0].neurons[2].inVal = row.second;

    layers[0].neurons[0].actVal = 1;
    layers[0].neurons[1].actVal = dataSet.normalizeX1(row.first);
    layers[0].neurons[2].actVal = dataSet.normalizeX2(row.second);

    // Hidden Layer
    for (size_t i = 1; i < 3; i++)
    {
      // Set input values
      layers[1].neurons[i].inVal = 0;
      cout << "size " << layers[0].neurons[i].weights.size() << endl;
      for (size_t j = 0; j < layers[1].neurons[i].weights.size(); j++)
      {
        layers[1].neurons[i].inVal += layers[0].neurons[j].actVal * layers[1].neurons[i].weights[j];
      }
      // Activate input values
      layers[1].neurons[i].actVal = logisticActivation(layers[1].neurons[i].inVal);
    }

    // Outer Layer
    for (size_t i = 0; i < 2; i++)
    {
      // Set input values
      layers[2].neurons[i].inVal = 0;
      for (size_t j = 0; j < layers[2].neurons[i].weights.size(); j++)
      {
        layers[2].neurons[i].inVal += layers[1].neurons[j].actVal * layers[2].neurons[i].weights[j];
      }
      // Activate input values
      layers[2].neurons[i].actVal = logisticActivation(layers[2].neurons[i].inVal);
    }
  }

  void backProp(pair<double, double> labels)
  {
    // Layer 2
    for (size_t i = 0; i < layers[2].neurons.size(); i++)
    {
      double actVal = layers[2].neurons[i].actVal;
      double expected;
      if (i == 0)
        expected = labels.first;
      else
        expected = labels.second;
      double G = LAMBDA * actVal * (1 - actVal) * (expected - actVal);
      layers[2].neurons[i].G = G;
      for (size_t j = 0; j < layers[2].neurons[i].weights.size(); j++)
      {
        layers[2].neurons[i].deltaWeights[j] = ETA * G * layers[1].neurons[j].actVal;
        layers[2].neurons[i].weights[j] = layers[2].neurons[i].weights[j] + layers[2].neurons[i].deltaWeights[j];
      }
    }
    // Layer 1
    for (size_t i = 1; i < layers[1].neurons.size(); i++)
    {
      double actVal = layers[1].neurons[i].actVal;
      cout << "layers[2].neurons[0].G: " << layers[2].neurons[0].G << endl;
      cout << "layers[2].neurons[0].weights[i]: " << layers[2].neurons[0].weights[i] << endl;
      cout << "layers[2].neurons[1].G : " << layers[2].neurons[1].G << endl;
      cout << "layers[2].neurons[0].weights[i]: " << layers[2].neurons[1].weights[i] << endl;

      double sum = layers[2].neurons[0].G * layers[2].neurons[0].weights[i];
      sum += layers[2].neurons[1].G * layers[2].neurons[1].weights[i];
      layers[1].neurons[i].sum = sum;

      double G = LAMBDA * actVal * (1 - actVal) * sum;
      layers[1].neurons[i].G = G;
      for (size_t j = 0; j < layers[1].neurons[i].weights.size(); j++)
      {
        layers[1].neurons[i].deltaWeights[j] = ETA * G * layers[0].neurons[j].actVal;
        layers[1].neurons[i].weights[j] = layers[1].neurons[i].weights[j] + layers[1].neurons[i].deltaWeights[j];
      }
    }
  }

  void train()
  {
    for (size_t i = 0; i < EPOCS; i++)
    {
      vector<pair<double, double>> features = dataSet.getFeatures("train", false);
      vector<pair<double, double>> normalizedFeatures = dataSet.getFeatures("train", true);
      vector<pair<double, double>> normalizedLabels = dataSet.getLabels("train", true);

      // for (size_t i = 0; i < features.size(); i++)
      for (size_t i = 0; i < 1; i++)
      {
        forwardProp(features[i]);
        printNetwork();
        cout << "----------" << endl;
        backProp(normalizedLabels[i]);
        printNetwork();
        cout << "Norm labels: " << normalizedLabels[i].first << " " << normalizedLabels[i].second << endl;
      }
    }
  }
};

int main()
{
  srand(42);
  Network network = Network(HIDDEN_SIZE);
  network.train();

  return 0;
}