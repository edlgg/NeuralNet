#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define DATA_LOCATION "dataNoNansNoDupsNoHeaders.csv"
// #define DATA_LOCATION "fakeData.csv"

#define IN_SIZE 2
#define HIDDEN_SIZE 2
#define OUT_SIZE 2

#define EPOCS 300
#define LAMBDA 1
#define ALPHA 0.6
#define ETA 0.8

#define CAP_DATA true
#define CAP_LIMIT 3000

#define INF 10000000

double r()
{
  double var = (double)rand() / (RAND_MAX);
  double value = (int)(var * 100 + .5);
  return (double)value / 100;
}

enum class LayerType
{
  Input,
  Hidden,
  Output
};

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
    this->data = {};
  }

  double y1Denorm(double val)
  {
    return val * (y1Max - y1Min) + y1Min;
  }

  double y2Denorm(double val)
  {
    return val * (y2Max - y2Min) + y2Min;
  }

  void readData()
  {
    string line;
    ifstream file(DATA_LOCATION);
    string x1, x2, y1, y2;
    if (file.is_open())
    {
      while (getline(file, x1, ','))
      {
        getline(file, x2, ',');
        getline(file, y1, ',');
        getline(file, y2, '\n');
        data.push_back({stod(x1), stod(x2), stod(y1), stod(y2)});
      }
      file.close();
    }
  }

  void computeMinAndMax()
  {
    x1Min = INF;
    x1Max = -1 * INF;
    x2Min = INF;
    x2Max = -1 * INF;
    y1Min = INF;
    y1Max = -1 * INF;
    y2Min = INF;
    y2Max = -1 * INF;
    for (size_t i = 0; i < data.size(); i++)
    {
      if (CAP_DATA)
      {
        if (data[i][0] > CAP_LIMIT)
          data[i][0] = CAP_LIMIT;
        if (data[i][1] > CAP_LIMIT)
          data[i][1] = CAP_LIMIT;
        if (data[i][2] > CAP_LIMIT)
          data[i][2] = CAP_LIMIT;
        if (data[i][3] > CAP_LIMIT)
          data[i][3] = CAP_LIMIT;
      }

      if (data[i][0] < x1Min)
        x1Min = data[i][0];
      if (data[i][0] > x1Max)
        x1Max = data[i][0];

      if (data[i][1] < x2Min)
        x2Min = data[i][1];
      if (data[i][1] > x2Max)
        x2Max = data[i][1];

      if (data[i][2] < y1Min)
        y1Min = data[i][2];
      if (data[i][2] > y1Max)
        y1Max = data[i][2];

      if (data[i][3] < y2Min)
        y2Min = data[i][3];
      if (data[i][3] > y2Max)
        y2Max = data[i][3];
    }
  }

  void printLimits()
  {
    cout << "x1Min: " << x1Min << endl;
    cout << "x1Max: " << x1Max << endl;
    cout << "x2Min: " << x2Min << endl;
    cout << "x2Max: " << x2Max << endl;
    cout << "y1Min: " << y1Min << endl;
    cout << "y1Max: " << y1Max << endl;
    cout << "y2Min: " << y2Min << endl;
    cout << "y2Max: " << y2Max << endl;
  }

  void normalizeData()
  {
    for (size_t i = 0; i < data.size(); i++)
    {
      data[i][0] = (data[i][0] - x1Min) / (x1Max - x1Min);
      data[i][1] = (data[i][1] - x2Min) / (x2Max - x2Min);
      data[i][2] = (data[i][2] - y1Min) / (y1Max - y1Min);
      data[i][3] = (data[i][3] - y2Min) / (y2Max - y2Min);
    }
  }

  void print()
  {
    for (size_t i = 0; i < data.size(); i++)
    {
      cout << data[i][0] << " " << data[i][1] << " " << data[i][2] << " " << data[i][3] << endl;
    }
    cout << "Data size: " << data.size() << endl;
    cout << "Train size: " << trainData.size() << endl;
    cout << "Validation size: " << validationData.size() << endl;
    cout << "Test size: " << testData.size() << endl;
  }

  void head()
  {
    for (size_t i = 0; i < 5; i++)
    {
      cout << data[i][0] << " " << data[i][1] << " " << data[i][2] << " " << data[i][3] << endl;
    }
  }

  void shuffleData()
  {
    return random_shuffle(data.begin(), data.end());
  }

  void shuffleTrainData()
  {
    return random_shuffle(trainData.begin(), trainData.end());
  }

  void splitData()
  {
    shuffleData();
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

  vector<vector<double>> getTrainFeatures()
  {
    vector<vector<double>> features = {};
    for (size_t i = 0; i < this->trainData.size(); i++)
    {
      // cout << "trainData[i][0]: " << trainData[i][0] << endl;
      features.push_back({this->trainData[i][0], trainData[i][1]});
    }
    return features;
  }

  vector<vector<double>> getTrainLabels()
  {
    vector<vector<double>> labels = {};
    for (size_t i = 0; i < trainData.size(); i++)
    {
      labels.push_back({trainData[i][2], trainData[i][3]});
    }
    return labels;
  }

  vector<vector<double>> getValidationFeatures()
  {
    vector<vector<double>> features = {};
    for (size_t i = 0; i < this->validationData.size(); i++)
    {
      features.push_back({this->validationData[i][0], validationData[i][1]});
    }
    return features;
  }

  vector<vector<double>> getValidationLabels()
  {
    vector<vector<double>> labels = {};
    for (size_t i = 0; i < validationData.size(); i++)
    {
      labels.push_back({validationData[i][2], validationData[i][3]});
    }
    return labels;
  }

  vector<vector<double>> getTestFeatures()
  {
    vector<vector<double>> features = {};
    for (size_t i = 0; i < this->testData.size(); i++)
    {
      features.push_back({this->testData[i][0], testData[i][1]});
    }
    return features;
  }

  vector<vector<double>> getTestLabels()
  {
    vector<vector<double>> labels = {};
    for (size_t i = 0; i < testData.size(); i++)
    {
      labels.push_back({testData[i][2], testData[i][3]});
    }
    return labels;
  }
};

struct Neuron
{
  double inVal;
  double actVal;
  double G;
  vector<double> weights;
  vector<double> deltaWeights;

  Neuron(int pastLayerSize)
  {
    this->inVal = 1;
    this->actVal = 1;
    this->G = 1;
    vector<double> weights = {};
    vector<double> deltaWeights = {};
    for (int i = 0; i < pastLayerSize; i++)
    {
      weights.push_back(r());
      deltaWeights.push_back(0);
    }
    this->weights = weights;
    this->deltaWeights = deltaWeights;
  }

  void computeActVal()
  {
    this->actVal = 1 / (1 + pow(2.7182, -1 * LAMBDA * this->inVal));
  }

  void computeDeltaWeights(LayerType layerType, double expectedVal, vector<double> previousActVals, double sum)
  {
    if (layerType == LayerType::Output)
    {
      double error = expectedVal - actVal;
      this->G = LAMBDA * actVal * (1 - actVal) * error;
    }
    else if (layerType == LayerType::Hidden)
    {
      this->G = LAMBDA * actVal * (1 - actVal) * sum;
    }

    for (size_t i = 0; i < deltaWeights.size(); i++)
    {
      deltaWeights[i] = ETA * this->G * previousActVals[i] + ALPHA * deltaWeights[i];
    }
  }

  void updateWeights()
  {
    for (size_t i = 0; i < weights.size(); i++)
    {
      weights[i] += deltaWeights[i];
    }
  }

  void printNeuron()
  {
    cout << "Neuron- inVal: " << inVal << " actVal: " << actVal << endl;
    cout << "   weights- ";
    for (size_t i = 0; i < weights.size(); i++)
    {
      cout << weights[i] << " ";
    }
    cout << endl;
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

  void setInVals(vector<double> vals) // Vals can be x (feature vector) or actVals (vector of activated values of past layer)
  {
    if (this->layerType == LayerType::Input)
    {
      for (int i = 1; i < neurons.size(); i++)
      {
        neurons[i].inVal = vals[i - 1];
      }
    }
    else if (this->layerType == LayerType::Hidden)
    {
      for (int i = 1; i < neurons.size(); i++)
      {
        double val = 0;
        for (int j = 0; j < vals.size(); j++)
        {
          // cout << "vals[j]: " << vals[j] << " neurons[i].weights[j]: " << neurons[i].weights[j] << endl;
          val += vals[j] * neurons[i].weights[j];
        }
        neurons[i].inVal = val;
      }
    }
    else
    {
      for (int i = 0; i < neurons.size(); i++)
      {
        double val = 0;
        for (int j = 0; j < vals.size(); j++)
        {
          val += vals[j] * neurons[i].weights[j];
        }
        neurons[i].inVal = val;
      }
    }
  }

  void activateNeurons()
  {
    for (size_t i = 0; i < neurons.size(); i++)
    {
      neurons[i].computeActVal();
    }
    if (layerType == LayerType::Input || layerType == LayerType::Hidden)
      neurons[0].actVal = 1;

    if (layerType == LayerType::Input){
      for (size_t i = 0; i < neurons.size(); i++)
        {
          neurons[i].actVal = neurons[i].inVal;
        }
    }
  }

  vector<double> getActVals()
  {
    vector<double> actVals = {};
    for (size_t i = 0; i < neurons.size(); i++)
    {
      actVals.push_back(neurons[i].actVal);
    }
    return actVals;
  }

  void forward(vector<double> x)
  {
    this->setInVals(x);
    this->activateNeurons();
  }

  void backward(vector<double> x, vector<double> previousActVals, vector<double> outLayerGs, vector<vector<double>> outWeights)
  {
    if (layerType == LayerType::Output)
    {
      for (int i = 0; i < neurons.size(); i++)
      {
        neurons[i].computeDeltaWeights(LayerType::Output, x[i], previousActVals, -1);
        neurons[i].updateWeights();
      }
    }
    else if (layerType == LayerType::Hidden)
    {
      for (int i = 0; i < neurons.size(); i++)
      {
        double sum = 0;

        for (size_t j = 0; j < OUT_SIZE; j++)
        {
          double a = outLayerGs[i];
          // cout << outWeights.size() << endl;
          double b = outWeights[j][i];
          sum += a + b;
        }
        neurons[i].computeDeltaWeights(LayerType::Hidden, x[i], previousActVals, sum);
        neurons[i].updateWeights();
      }
    }
  }

  vector<double> getGs()
  {
    vector<double> Gs = {};
    for (size_t i = 0; i < neurons.size(); i++)
    {
      Gs.push_back(neurons[i].G);
    }
    return Gs;
  }

  vector<vector<double>> getWeights()
  {
    vector<vector<double>> weights = {};
    for (size_t i = 0; i < neurons.size(); i++)
    {
      weights.push_back(neurons[i].weights);
    }
    return weights;
  }

  void printLayer()
  {
    cout << "Layer" << endl;
    for (size_t i = 0; i < neurons.size(); i++)
    {
      neurons[i].printNeuron();
    }
    cout << endl
         << endl;
  }
};

struct Network
{
  DataSet dataSet;
  vector<Layer> layers;

  Network()
  {
    this->dataSet = DataSet();
    Layer inputLayer = Layer(LayerType::Input, IN_SIZE, 0);
    Layer hiddenLayer = Layer(LayerType::Hidden, HIDDEN_SIZE, IN_SIZE + 1);
    Layer outLayer = Layer(LayerType::Output, OUT_SIZE, HIDDEN_SIZE + 1);
    this->layers = {inputLayer, hiddenLayer, outLayer};
  }

  void forwardProp(vector<double> x)
  {
    this->layers[0].forward(x);
    this->layers[1].forward(this->layers[0].getActVals());
    this->layers[2].forward(this->layers[1].getActVals());
  }

  void backProp(vector<double> x)
  {
    vector<double> previousActVals = this->layers[1].getActVals();
    this->layers[2].backward(x, previousActVals, {}, {{}});
    previousActVals = this->layers[0].getActVals();
    vector<double> outLayerGs = this->layers[2].getGs();
    vector<vector<double>> outWeights = this->layers[2].getWeights();
    this->layers[1].backward(x, previousActVals, outLayerGs, outWeights);
  }

  void printNetwork()
  {
    for (size_t i = 0; i < layers.size(); i++)
    {
      layers[i].printLayer();
    }
  }
  pair<double, double> predict(vector<double> row)
  {
    this->forwardProp(row);
    double y1 = layers[2].neurons[0].actVal;
    double y2 = layers[2].neurons[1].actVal;
    // double y1ActualDenorm = dataSet.y1Denorm(y.first);
    // double y2ActualDenorm = dataSet.y2Denorm(y.second);
    return make_pair(y1, y2);
  }

  double getMSE()
  {
    vector<vector<double>> testFeatures = dataSet.getValidationFeatures();
    vector<vector<double>> testLabels = dataSet.getValidationLabels();
    double mse = 0;
    for (size_t i = 0; i < testFeatures.size(); i++)
    {
      pair<double, double> y = predict(testFeatures[0]);
      double y1ActualDenorm = dataSet.y1Denorm(y.first);  // TODO: move this to predict
      double y2ActualDenorm = dataSet.y2Denorm(y.second); // TODO: Have to add load limits to loadModel
      double y1ExpectedDenorm = dataSet.y1Denorm(testLabels[i][0]);
      double y2ExpectedDenorm = dataSet.y2Denorm(testLabels[i][1]);
      double y1SError = pow(y1ExpectedDenorm - y.first, 2);
      double y2SError = pow(y2ExpectedDenorm - y.second, 2);
      mse += (y1SError + y2SError) / 2;
    }
    return mse / testFeatures.size();
  }

  double getME()
  {
    vector<vector<double>> testFeatures = dataSet.getValidationFeatures();
    vector<vector<double>> testLabels = dataSet.getValidationLabels();
    double mse = 0;
    for (size_t i = 0; i < testFeatures.size(); i++)
    {
      pair<double, double> y = predict(testFeatures[0]);
      double y1ExpectedDenorm = dataSet.y1Denorm(testLabels[i][0]);
      double y1ActualDenorm = dataSet.y1Denorm(y.first);
      double y2ExpectedDenorm = dataSet.y2Denorm(testLabels[i][1]);
      double y2ActualDenorm = dataSet.y2Denorm(y.second);
      double y1SError = abs(y1ExpectedDenorm - y1ActualDenorm);
      double y2SError = abs(y2ExpectedDenorm - y1ActualDenorm);
      mse += (y1SError + y2SError) / 2;
    }
    return mse / testFeatures.size();
  }

  void saveState()
  {
    ofstream file;
    string hiddenSize = to_string(layers[1].neurons.size() - 1);
    file.open("savedModel_" + hiddenSize + ".txt");
    double hiddenLayerSize = layers[1].neurons.size();
    file << hiddenLayerSize << "\n";
    for (size_t i = 1; i < 3; i++)
    {
      for (size_t j = 0; j < layers[i].neurons.size(); j++)
      {
        // cout << "layers[i].neurons.size(): " << layers[i].neurons.size() << endl;

        for (size_t k = 0; k < layers[i].neurons[j].weights.size(); k++)
        {
          // cout << "layers[i].neurons[j].weights.size(): " << layers[i].neurons[j].weights.size() << endl;
          file << layers[i].neurons[j].weights[k] << "\n";
        }
        file << "\n";
      }
      file << "\n";
    }
    file.close();
  }

  void train()
  {
    dataSet.readData();
    dataSet.computeMinAndMax();
    dataSet.printLimits();
    dataSet.normalizeData();
    dataSet.splitData();

    for (size_t i = 0; i < EPOCS; i++)
    for (size_t i = 0; i < 1; i++)
    {
      cout << "EPOC: " << i << endl;
      dataSet.shuffleTrainData();
      for (size_t j = 0; j < dataSet.trainData.size(); j++)
      for (size_t j = 0; j < 1; j++)
      {
        vector<double> row = dataSet.getTrainFeatures()[j];
        // cout << "ROW " << row[0] << " : " << row[1] << endl;
        // cout << "1--" << endl;
        // printNetwork();
        forwardProp(row);
        // cout << "2--" << endl;
        // printNetwork();
        // cout << "3--" << endl;
        backProp(row);
        // printNetwork();
      }
      cout << "  MSE: " << getMSE() << endl;
      if (i % 5 == 0)
        saveState();
      cout << "  ME: " << getME() << endl;
    }
  }

  void loadWeights()
  {
    ifstream file;
    file.open("savedModel_9.txt");
    string x;
    file >> x;
    int temp = stoi(x);
    for (int i = 0; i < temp; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        file >> x;
        layers[1].neurons[i].weights[j] = stod(x);
      }
    }
    for (int i = 0; i < 2; i++)
    {
      for (int j = 0; j < temp; j++)
      {
        file >> x;
        layers[2].neurons[i].weights[j] = stod(x);
      }
    }
    file.close();
  }

  pair<double, double> evaluate(double x1, double x2)
  {
    loadWeights();
    pair<double, double> resp = predict({x1, x2});
    double y1 = resp.first * (224 - 115) + 15;
    double y2 = resp.second * (300 - 81) + 81;
    return make_pair(y1, y2);
  }

  void test(){
    dataSet.readData();
    dataSet.computeMinAndMax();
    dataSet.printLimits();
    dataSet.normalizeData();
    dataSet.splitData();

    vector<vector<double>> features = this->dataSet.getTestFeatures();
    vector<vector<double>> labels = this->dataSet.getTestLabels();
    cout << features.size();
    cout << "aaa" << features[1][0] << " "<<features[1][1] << endl;
    for (size_t i = 0; i < features.size(); i++)
    {
      pair<double,double> Y = this->evaluate(features[i][0],features[i][1]);
      cout << "x1: " << features[i][0] << " x2: " << features[i][1] << " y1e: " << labels[i][0] << " y2e: " << labels[i][1] << " y1a: " << Y.first << " y2a: " << Y.second << endl;
    }


  }
};

int main()
{
  srand(42);

  Network network = Network();
  network.train();
  // network.evaluate(1880, 1480); //115 300
  // network.evaluate(480, 2420); //115 123
  // network.test();

  return 0;
}
