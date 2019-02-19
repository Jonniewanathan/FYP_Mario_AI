from multiprocessing import Process
import os


list_of_names = ['john', 'bob', 'frank']


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(name):
    info('function f')
    print('hello', name)


if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=(list_of_names,))
    p.start()
    p.join()
