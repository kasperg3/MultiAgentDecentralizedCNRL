//
// Created by kasper on 10/27/19.
//

#ifndef MERGABLE_INDUSTRIAL_ROBOTS_TTTRL_H
#define MERGABLE_INDUSTRIAL_ROBOTS_TTTRL_H

class TTTRL {

public:
    TTTRL();
    ~TTTRL();



private:

    double gameState[9];
    int gameCount;

    int gameFinished();
    double evalQ();
    void makeMove();
    void playGame();
    void learn();

};


#endif //MERGABLE_INDUSTRIAL_ROBOTS_TTTRL_H
