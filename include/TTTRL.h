//
// Created by kasper on 10/27/19.
//

#ifndef MERGABLE_INDUSTRIAL_ROBOTS_TTTRL_H
#define MERGABLE_INDUSTRIAL_ROBOTS_TTTRL_H
#include <vector>
#include <iostream>
#include <map>
#include <algorithm>
#include "../lib/matplotlib/matplotlibcpp.h"

namespace plt = matplotlibcpp;

class TTTRL {

public:
    TTTRL();
    int playGame();

    void playGames(int);
private:
    std::vector<int> gameState;
    std::vector<std::vector<int>> qPlayerStateHistory;
    void resetGameState();
    int gameFinished();
    void makeRandomMove(int);
    void makeQmove(int);
    bool isValidMove(int);
    void printGameState();
    double getMaxElement(std::vector<double>);
    int getMaxElementIndex(std::vector<double>);
    void rewardQPlayer(int player, int won);
    void printVector(std::vector<double> const &input);
    std::vector<double> getAndCreateQVector(std::vector<int>);

    int player1 = 1;
    int player2 = 2;
    std::map<std::vector<int>, std::vector<double>> qTable;

    //Q LEARNING DEFINES
    double learningRate = 0.90;
    //double discountFactor = 0.95;
    std::vector<double> discountFactor = {0.05, 0.2, 0.35, 0.45, 0.65, 0.75, 0.8, 0.9, 0.95};
    //std::vector<double> discountFactor = {0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95};

};


#endif //MERGABLE_INDUSTRIAL_ROBOTS_TTTRL_H
