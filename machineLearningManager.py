import linearRegression, KNearestAlg, smvWorking, kMeans
import time


class manager:
    def main(self):
        print("-----------------------------------------")
        print("------- Machine Learning Manager --------")
        print("--- Developed by Jrbyte on github.com ---")
        print("-----------------------------------------")

        while True:
            print("(1): Linear Regression, purpose: regression")
            print("(2): K Nearest Algorithm, purpose: classification & regression")
            print("(3): SMV Algorithm, purpose: classification & regression")
            print("(4): K Means Clustering Algorithm, purpose: classification & regression")
            value = input("\n[?]: Enter a number for which algorithm you want to run: ")
            if value == "1":
                print("\n[Running]: Linear Regression")
                time.sleep(2)
                linearRegression.example1().main()
                break
            elif value == "2":
                print("\n[Running]: K Nearest Algorithm")
                time.sleep(2)
                KNearestAlg.example1().main()
                break
            elif value == "3":
                print("\n[Running]: SMV")
                time.sleep(2)
                smvWorking.example1().main()
                break
            elif value == "4":
                print("\n[Running]: K Means Algorithm")
                time.sleep(2)
                kMeans.example1().main()
                break
            # elif value == "5":
            #     print("\n[Running]: ?????")
            #     time.sleep(2)
            #     print("N/A")
            #     break
            else:
                print("\n[!]: Wrong value printed... Try again.\n")
                continue


if __name__ == '__main__':
    manager().main()
