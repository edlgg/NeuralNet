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

#define EPOCS 1000
#define LAMBDA 1
#define ALPHA 0.6
#define ETA 1

#define CAP_DATA false
#define CAP_LIMIT 3000

#define INF 10000000

enum class LayerType
{
  Input,
  Hidden,
  Output
};

struct DataSet{
  vector<vector<double>> data;
  double x1Min;
  double x1Max;
  double x2Min;
  double x2Max;
  double y1Min;
  double y1Max;
  double y2Min;
  double y2Max;

  DataSet(){
    data = {};
    x1Min = INF;
    x1Max = -1 * INF;
    x2Min = INF;
    x2Max = -1 * INF;
    y1Min = INF;
    y1Max = -1 * INF;
    y2Min = INF;
    y2Max = -1 * INF;
  }

  void readData(string fileName){
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
        if(stod(x1) > x1Max) x1Max = stod(x1);
        if(stod(x1) < x1Min) x1Min = stod(x1);
        if(stod(x2) > x2Max) x2Max = stod(x2);
        if(stod(x2) < x2Min) x2Min = stod(x2);
        if(stod(y1) > y1Max) y1Max = stod(y1);
        if(stod(y1) < y1Min) y1Min = stod(y1);
        if(stod(y2) > y2Max) y2Max = stod(y2);
        if(stod(y2) < y2Min) y2Min = stod(y2);
      }
      file.close();
    }
  }
  double normalizeX1(double x1){
    return (x1-x1Min)/(x1Max-x1Min);
  }
  double normalizeX2(double x2){
    return (x2-x2Min)/(x2Max-x2Min);
  }
  double normalizeY1(double y1){
    return (y1-y1Min)/(y1Max-y1Min);
  }
  double normalizeY2(double y2){
    return (y2-y2Min)/(y2Max-y2Min);
  }
  double denormalizeY1(double y1pred){
    return y1pred*(y1Max-y1Min)+y1Min;
  }
  double denormallizeY2(double y2pred){
    return y2pred*(y2Max-y2Min)+y2Min;
  }
  vector<pair<double,double>> getFeatures(bool normalized){
    vector<pair<double,double>> features = {};
    for (size_t i = 0; i < this->data.size(); i++)
    {
      pair<double, double> p = make_pair(data[i][0], data[i][1]);
      if(normalized){
        p.first = this->normalizeX1(p.first);
        p.second = this->normalizeX2(p.second);
      }
      features.push_back(make_pair(p.first,p.second));
    }
    return features;
  }
  vector<pair<double,double>> getLabels(bool normalized){
    vector<pair<double,double>> labels = {};
    for (size_t i = 0; i < this->data.size(); i++)
    {
      pair<double, double> p = make_pair(data[i][2], data[i][3]);
      if(normalized){
        p.first = this->normalizeY1(p.first);
        p.second = this->normalizeY2(p.second);
      }
      labels.push_back(make_pair(p.first,p.second));
    }
    return labels;
    }
    void shuffle(){
      return random_shuffle(data.begin(), data.end());
    }
    void info(){
      cout << "data.size(): " << data.size() << endl;
      cout << "x1Min: " << x1Min << "  x1Max: " << x1Max << endl;
      cout << "x2Min: " << x2Min << "  x2Max: " << x2Max << endl;
      cout << "y1Min: " << y1Min << "  y1Max: " << y1Max << endl;
      cout << "y2Min: " << y2Min << "  y2Max: " << y2Max << endl;
      int count = fmin(5,data.size());
      for(int i=0;i<count;i++){
        cout <<data[i][0]<< " "<<data[i][1]<< " "<<data[i][2]<< " "<<data[i][3]<< endl;
      }
    }

};

struct Neuron
{
  double inVal;
  double actVal;
  double G;
  vector<double> weights;
  vector<double> deltaWeights;

  // double logisticActivation(double val){}
  // vector<double> computeAndSetDeltaWeights(LayerType layerType, double expectedVal, vector<double> previousActVals, double sum){}
};

int main()
{
  srand(5);

  DataSet dataSet = DataSet();
  dataSet.info();
  dataSet.readData(DATA_LOCATION);
  cout << "********" << endl;
  dataSet.info();
  return 0;
}