all:
	rm -rf bin
	mkdir bin
	make -C ipac_src/ipac_multi
	make -C ipac_src/ipac_single
	cp -rf ipac_src/ipac_multi/output/bin/ipac ./bin/ipac_multiple
	cp -rf ipac_src/ipac_single/output/bin/ipac ./bin/ipac_single

clean:
	rm -rf bin
	make -C ipac_src/ipac_multi clean
	make -C ipac_src/ipac_single clean
