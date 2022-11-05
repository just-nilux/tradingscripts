#!/bin/python3
import sys

def main(*args, **kwargs):
    """
    Hej, Liam.
    Denne lille fil viser nogle typiske eksempler, på ting
    der er fine at vide når man vil lave Python Scripts i Linux.

    Jeg synes du skal åbne din yndlings editor og kigge lidt
    rundt i den.

    Du kan kalde den med alle argumenter du skulle have lyst til,
    eks:
    python3 args.py arg1 arg2 kw1=hello kw2=world
    
    Og faktisk kan du blot skrive:
    ./args.py arg1 arg2 kw1=hello kw2=world

    Det skyldes linje 1 i denne fil:
    #!/bin/python3

    Det kaldes en Shebang (hehe). Den fortæller Linux hvordan
    filen skal læses. På Linux gemmes programmer det samme sted,
    nemlig i mappen /bin - og uha, forestil dig, man ville gøre 
    samme trick i Windows: Gys! 😱

    Man skal selvfølgelig huske at gøre filen eksekverbar:
    sudo chmod +x args.py

    Men prøv nu at kalde filen med nogle argumenter!
    """

    if len(args) > 0:
        print("""
    Sikke nogle fine argumenter, min ven! Dem kan jeg ikke diskutere
    med. Lad mig lige gentage dem:
        """)
        for index, arg in enumerate(args):
            print("arg", f"{index} er", arg)

    if len(kwargs) > 0:
        print("""
    Og med navn! Sikke en fest. Tænk nu, hvis det var filnavne,
    navn på crypto trading pair, eller en vigtig konstant?
    """)
        for key, value in kwargs.items():
            print(f"kwarg['{key}'] =", value)


if __name__=="__main__":
    # sys.argv[0] er altid lig kaldet der kørte scriptet. Ex:
    # sys.argv[0] == "python3 arg.py"
    if len(sys.argv) == 1:
        # Hvor kommer __doc__ mon fra? Spændende at forske i!
        print(main.__doc__)

    else:
        # Hvis vi havde flere argumenter...
        args = []
        kwargs = {}
        for arg in sys.argv[1:]:
            if "=" in arg:
                key, value = arg.split("=")
                kwargs[key] = value
            else:
                args.append(arg)

        main(*args, **kwargs)
