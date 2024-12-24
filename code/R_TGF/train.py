# -*- coding: utf-8 -*-

'''
程序入口
'''

from FINDER import FINDER

def main():
    dqn = FINDER()
    dqn.Train()


if __name__=="__main__":
    main()