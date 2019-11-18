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

#define EPOCS 1000
#define LAMBDA 1
#define ALPHA 0.6
#define ETA 1

#define CAP_DATA false
#define CAP_LIMIT 3000

#define INF 10000000

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

enum class LayerType
{
  Input,
  Hidden,
  Output,
  None
};

struct Neuron
{
  LayerType layerType;
  int i;
  double inVal;
  double actVal;
  double G;
  vector<double> weights;
  vector<double> deltaWeights;

  Neuron(LayerType layerType, int i)
  {
    inVal = 1;
    actVal = 1;
    G = 1;
    vector<double> w = {};
    vector<double> dw = {};
    for (int i = 0; i < pastLayerSize; i++)
    {
      w.push_back(this->r());
      dw.push_back(0);
    }
    weights = w;
    deltaWeights = dw;
  }

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
  // vector<vector<double>> computeDeltaWeights(LayerType layerType, double expectedVal, vector<double> previousActVals, double sum) {}

  computeInVal(Network network, pair<double, double> x)
  {
    vector<double> actVals;
    if (layerType == LayerType::Input)
    {
      actVals = row;
    }
    else
    {
      // if (layerType == LayerType::Hidden)
      // {
      //   actVals = network.layers[0].getActVals();
      // }
      // else if (layerType == LayerType::Output)
      // {
      //   actVals = network.layers[1].getActVals();
      // }
      // else
      //   exit(1);
      // if (actVals.size() != weights.size())
      //   exit(1);
      // double inVal = 0;
      // for (size_t i = 0; i < weights.size(); i++)
      // {
      //   inVal += actVals[i] * weights[i];
      // }
    }
    return inVal;
  }

  computeActVal()
  {
    if (layerType == LayerType::Input)
  }
};

struct Layer
{
  LayerType layerType;
  vector<Neuron> neurons;

  Layer(LayerType layerType, int layerSize, int pastLayerSize)
  {
    this->layerType = layerType;
    vector<Neuron> neurons = {};
    if (layerType == LayerType::Input || layerType == LayerType::Hidden)
      neurons.push_back(Neuron(pastLayerSize));
    for (int i = 0; i < layerSize; i++)
      neurons.push_back(Neuron(pastLayerSize));
    this->neurons = neurons;
  }

  vector<double> getActVals()
  {
    vector<double> actVals = {} for (size_t i = 0; i < neurons.size(); i++)
    {
      actVals.push_back(neurons[i].actVal);
    }
    return actVals;
  }
};

// struct Network
// {
//   DataSet dataSet;
//   vector<Layer> layers;

//   Network(){
//     this->dataSet = DataSet DataSet();
//     this->layers = ;
//   }
// };

int main()
{
  srand(5);

  DataSet dataSet = DataSet();
  dataSet.info();
  dataSet.readData(DATA_LOCATION);
  cout << "**" << endl;
  dataSet.info();
  dataSet.shuffle("all");
  cout << "****" << endl;
  dataSet.info();
  dataSet.splitData();
  cout << "******" << endl;
  dataSet.info();
  cout << "**********" << endl;
  vector<pair<double, double>> features = dataSet.getFeatures("validation", true);
  for (size_t i = 0; i < 5; i++)
  {
    cout << features[i].first << endl;
  }

  return 0;
}