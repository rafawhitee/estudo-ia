valor=float(input("Digite o valor da casa:")) # 1 
salario=float(input("Digite o salário:")) # 2
anos=float(input("Quantos anos para pagar:")) # 3 e 8
meses = anos * 12
prestacao = valor / meses
if prestacao > (salario * 0.3): #4
	print("Infelizmente você não pode obter o empréstimo")
else: # 5
    print(f"Valor da prestação: R$ {prestacao:.2f} Empréstimo OK" ) # 6, 7
