
//
// Created by kasper on 10/27/19.
//
#include "TTTRL.h"


TTTRL::TTTRL() {
    //Init game parameters
    resetGameState();
    srand(time(nullptr));

    //Init q table
}


void TTTRL::resetGameState() {
    //Init gamestates
    gameState = {0,0,0,
                 0,0,0,
                 0,0,0};

    qPlayerStateHistory = {};

}

/*
 * Return 0 if not finished, if player 1 won, return 1, 2 if player 2 won
 * return -1 if tied
 */


int TTTRL::gameFinished() {
    std::vector<int> checkArray1 = {0,1,2,0,2,0,3,6};
    std::vector<int> checkArray2 = {3,4,5,4,4,1,4,7};
    std::vector<int> checkArray3 = {6,7,8,8,6,2,5,8};

    for(int i = 0; i < checkArray1.size(); i++){
        //Check if player 1 has 3 in a row
        if(gameState[checkArray1[i]] == player1 && gameState[checkArray2[i]] == player1 && gameState[checkArray3[i]] == player1){
            return 1;
        }
        //Check if player 2 has 3 in a row
        if(gameState[checkArray1[i]] == player2 && gameState[checkArray2[i]] == player2 && gameState[checkArray3[i]] == player2){
            return 2;
        }
    }

    //Checks if there are more vacant spots left
    int vacantSpots = 0;
    for(int i = 0; i < gameState.size(); i++){
        if(gameState[i] == 0)
            vacantSpots++;
    }
    if(vacantSpots == 0)
        return -1;
    else return 0;
}

void TTTRL::printGameState() {

    std::cout << gameState[0]  << " " << gameState[1]  << " " << gameState[2]   << std::endl;
    std::cout << gameState[3]  << " " << gameState[4]  << " " << gameState[5]   << std::endl;
    std::cout << gameState[6]  << " " << gameState[7]  << " " << gameState[8]   << std::endl;

    std::cout << std::endl << std::flush;
}

bool TTTRL::isValidMove(int idx) {
    if(gameState[idx] != 0)
        return false;
    else
        return true;
}

void TTTRL::makeRandomMove(int player) {

    if(gameFinished()){
        return;
    }
    int randIdx = rand()%9;
    while(!isValidMove(randIdx)){
        randIdx = rand()%9;
    }
    gameState[randIdx] = player;
}



int TTTRL::playGame() {
    resetGameState();
    //printGameState();
    while(gameFinished() == 0){
        //printGameState();
        makeRandomMove(player1);
        makeQmove(player2);
        //printGameState();
    }
    //printGameState();
    //std::cout << "the winner is: " << gameFinished() << std::endl;

    //Evaluate the Q player and give reward
    rewardQPlayer(player2, gameFinished());

    return gameFinished();
}

void TTTRL::playGames(int numberGames) {
    double player1Wins = 0;
    double player2Wins = 0;
    double ties = 0;
    std::vector<double> playerTies = {};
    std::vector<double> playerQWins = {};
    std::vector<double> playerRandWins = {};
    std::vector<double> xAxis = {};


    for(int i = 0; i < numberGames; i++){
        if(playGame()==-1){
            ties += 1;
        }else if(playGame()==1){
            player1Wins+= 1;
        }else if(playGame()==2){
            player2Wins+= 1;
        }

        std::cout
        << "Qplayer[%] " << (double)(player2Wins/(i+1))*100.0
        << " Random[%]: " << (double)(player1Wins/(i+1))*100.0
        <<" Ties[%]: " << (double)(ties/(i+1))*100.0
        << " WinRatio[P1/P2]: "  << player1Wins << "/" << player2Wins
        << " Total games: " << numberGames << std::endl;

        playerQWins.push_back((double)(player2Wins/(i+1))*100.0);
        playerRandWins.push_back((double)(player1Wins/(i+1))*100.0 );
        playerTies.push_back((double)(ties/(i+1))*100.0 );
        xAxis.push_back(i);
    }

    plt::figure_size(1200, 780);
    plt::named_plot("Q player[%]",xAxis, playerQWins);
    plt::named_plot("Random player[%]", xAxis, playerRandWins);
    plt::named_plot("Ties[%]", xAxis, playerTies);
    plt::ylim(0, 100);
    std::string plotTitle("Q learner starting first");
    plt::title(plotTitle);
    plt::legend();
    plt::show();
    plt::save(plotTitle.append(".png"));

}

std::vector<double> TTTRL::getAndCreateQVector(std::vector<int> state){
    std::vector<double> qValues;
    if(qTable.find(state) == qTable.end()){
        //Not found
        //Check if there is a valid "move", then initialize it to 0.1, if invalid init to 0
        for(int j = 0; j < gameState.size(); j++){
            if(state[j] == 0){
                double randInit = ((rand() % 1000) + 1)/1000.0;
                qValues.push_back(randInit);
            }else
                qValues.push_back(0);
        }

        qTable[state] = qValues;
        return qTable[state];
    }
    return qTable[state];
}

void TTTRL::rewardQPlayer(int player, int won) {
    std::vector<double> winningQValues = getAndCreateQVector(gameState);

    double reward;
    if(won==player1){
        reward = 0;
    }
    if(won == player2){
        reward = 1;
    }
    if(won == -1){
        reward = 0.5;
    }

    for(int j = 0; j < qPlayerStateHistory.size()-1; j++){
        std::vector<double> currentQValues = getAndCreateQVector(qPlayerStateHistory[j]);
        double maxQValueNextState = getMaxElement(getAndCreateQVector(qPlayerStateHistory[j+1]));
        std::vector<double> newQValue = {};
        for(int i = 0; i < winningQValues.size(); i++){
            if(getAndCreateQVector(qPlayerStateHistory[j])[i] != 0){
                double qValue = currentQValues[i] + learningRate*(discountFactor[i]*maxQValueNextState-currentQValues[i]);
                newQValue.push_back(qValue);
            }else{
                newQValue.push_back(0);
            }
        }
        qTable[qPlayerStateHistory.at(j)] = newQValue;
    }
}

void TTTRL::makeQmove(int player) {
    qPlayerStateHistory.push_back(gameState); //Save the current gamestate
    //Check that the game is not finished
    if(gameFinished()){
        //Already lost, because starting number 2
        return;
    }
    //get State from q table and choose max value move
    int bestMove = getMaxElementIndex(getAndCreateQVector(gameState));
    gameState[bestMove] = player;
}

void TTTRL::printVector(std::vector<double> const &input){
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(i) << ' ';
    }
    std::cout << std::endl;
}

double TTTRL::getMaxElement(std::vector<double> input) {
    return *std::max_element(input.begin(), input.end());
}

int TTTRL::getMaxElementIndex(std::vector<double> input) {
    auto maxi = std::max_element(input.begin(), input.end());
    return std::distance(input.begin(), maxi);
}



