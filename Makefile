CFLAGS=
FILE_NAME=xor

all:
	cl65 $(CFLAGS) -Oi $(FILE_NAME).c -o $(FILE_NAME).prg
	rm -f $(FILE_NAME)_disk.d64
	c1541 -format diskname,id d64 $(FILE_NAME)_disk.d64 -attach $(FILE_NAME)_disk.d64 -write $(FILE_NAME).prg $(FILE_NAME)	

disk:
	rm -f $(FILE_NAME)_disk.d64
	c1541 -format diskname,id d64 $(FILE_NAME)_disk.d64 -attach $(FILE_NAME)_disk.d64 -write $(FILE_NAME).prg $(FILE_NAME)

clean:
	rm -f $(FILE_NAME).prg $(FILE_NAME)_disk.d64 *~
