import numpy as np
from . import TicTacToe
from . import backwardInduction
from time import sleep


def userChoice(query, default=True):
    if default:
        choices = 'Y/n'
        defChoice = 'Y'
    else:
        choices = 'y/N'
        defChoice = 'N'

    answer = input(query + '[' + choices + '] ').upper()
    while not (len(answer) == 0 or answer[0] in ('', 'N', 'Y')):
        answer = input("Sorry, I didn't get that. Say again!\n--> ").upper()

    if len(answer) == 0:
        c = defChoice
    else:
        c = answer[0]

    if c == 'Y':
        return True
    return False


def play():
    game = TicTacToe()

    name = input('Hello Tic Tac Toe player!\n'
                 'What is your name, sweetheart?\n'
                 '--> ')
    print('Uuuuh, ' + name + ' is a beautiful name!')

    userFirst = userChoice('So, ' + name + ', do you want to go first?\n'
                           '--> ', False)
    if userFirst:
        user = TicTacToe.P1
    else:
        user = TicTacToe.P2

    while not game.inTerminalState():
        if game.getPlayer() == user:
            print(game)
            action = ''
            while not game.checkAction(action):
                try:
                    action = int(input(name + '! Which action?\n--> '))
                except ValueError:
                    print("Oops, that doesn't make sense...")
        else:
            # TODO: implement backward induction
            _, bestActions = backwardInduction(game)
            action = bestActions[0]

            #print(game)
            #action = ''
            #while not game.checkAction(action):
            #    try:
            #        action = int(input('Computer' + '! Which action?\n--> '))
            #    except ValueError:
            #        print("Oops, that doesn't make sense...")

        game.performAction(action)

    print("\nIt's over! Let's see who has won...")
    sleep(1.)
    print(game)
    sleep(1.)
    _, winner = game.getTerminalStateAndWinner()
    if winner == user:
        print('Yay, you made it!')
        sleep(1.)
        print('... wait ....')
        sleep(1.)
        print("look's like your algorithm is still a bit buggy, honey!")
    elif winner == 0:
        print('A draw! Again? How boring...')
    else:
        print('You lost.')

    sleep(1.)
    print('\n\n\n')
