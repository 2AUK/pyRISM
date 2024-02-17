import tabula

tabula.convert_into("data.pdf", "output_lattice.csv", output_format="csv", pages='all', lattice=True)
