class A:
    def __init__(self, a):
        self.a = a
        self.echo()
    def echo(self):
        print(self.a)
if __name__ == "__main__":
    aa = A(10)