import linearRegression, KNearestAlg
import time


class manager:
    def main(self):
        print("-----------------------------------------")
        print("------- Machine Learning Manager --------")
        print("--- Developed by Jrbyte on github.com ---")
        print("-----------------------------------------")

        while True:
            print("(1): Linear Regression")
            print("(2): K Nearest Algorithm")
            value = input("[?]: Enter an 1 or 2 for which algorithm you want to run: ")
            if value == "1":
                print("\n[Running]: Linear Regression")
                time.sleep(2)
                linearRegression.test1().main()
                break
            elif value == "2":
                print("\n[Running]: K Nearest Algorithm")
                time.sleep(2)
                KNearestAlg.firstTest().main()
                break
            else:
                print("\nWrong value printed... Try again.\n")
                continue


if __name__ == '__main__':
    manager().main()
