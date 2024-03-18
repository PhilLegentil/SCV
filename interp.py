import pandas as pd
import numpy as np
import csv
import win32com.client
import os

logf = open("errorlog.txt", "w")

try:

    print("Interpolation")

    with open('donneinterp.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        titre = "".join(next(reader))
        W_net = next(reader)
        rendement = next(reader)

    for i, w in enumerate(W_net):
        W_net[i] = float(w)

    for i, r in enumerate(rendement):
        rendement[i] = float(r)

    file_path = "../interface.xlsm"
    data = pd.read_excel(file_path, header=0, sheet_name=titre)
    x = data['interp'][0]

    valeur = np.interp(x, W_net, rendement)

    ExcelApp = win32com.client.GetActiveObject("Excel.Application")
    ExcelApp.Visible = True

    workbook = ExcelApp.Workbooks.Open(r"../interface.xlsm")
    sheet = workbook.Worksheets(2)

    sheet.Range("X3").Value = valeur

    os.chdir('C:/Users/Phil')

except Exception as e:  # most generic exception you can catch
    logf.write(str(e))
finally:
    pass
