ifeq ($(shell uname -m),x86_64)
#CXXDIR="/home/eeyang/yorkey/usr/gcc_4.8.2/bin/"
CXXDIR=""
CC="${CXXDIR}gcc"
CXX="${CXXDIR}g++"
CXXFLAGS=-g -O2 \
  -pipe \
  -W \
  -Wall \
  -fPIC -fopenmp
CFLAGS=-g -O2 \
  -pipe \
  -W \
  -Wall \
  -fPIC -fopenmp
CPPFLAGS=-D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\"
INCPATH=-I. \
  -I./include
DEP_INCPATH=


.PHONY:all
all:ipac libipac.a libipac.so 
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40mall[0m']"
	@echo "make all done"

.PHONY:ccpclean
ccpclean:
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40mccpclean[0m']"
	@echo "make ccpclean done"

.PHONY:clean
clean:ccpclean
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40mclean[0m']"
	rm -rf ipac
	rm -rf ./output/bin/ipac
	rm -rf libipac.a
	rm -rf ./output/lib/libipac.a
	rm -rf ./output/include/data.h
	rm -rf ./output/include/dmatrix.h
	rm -rf ./output/include/io.h
	rm -rf ./output/include/io_utils.h
	rm -rf ./output/include/ipac.h
	rm -rf ./output/include/iterator.h
	rm -rf ./output/include/matrix_csr.h
	rm -rf ./output/include/omp_xg.h
	rm -rf ./output/include/page_dmatrix-inl.hpp
	rm -rf ./output/include/page_fmatrix-inl.hpp
	rm -rf ./output/include/random.h
	rm -rf ./output/include/simple_dmatrix-inl.hpp
	rm -rf ./output/include/simple_fmatrix-inl.hpp
	rm -rf ./output/include/thread_buffer.h
	rm -rf ./output/include/thread.h
	rm -rf ./output/include/utils.h
	rm -rf libipac.so
	rm -rf ./output/so/libipac.so
	rm -rf ./output/include/data.h
	rm -rf ./output/include/dmatrix.h
	rm -rf ./output/include/io.h
	rm -rf ./output/include/io_utils.h
	rm -rf ./output/include/ipac.h
	rm -rf ./output/include/iterator.h
	rm -rf ./output/include/matrix_csr.h
	rm -rf ./output/include/omp_xg.h
	rm -rf ./output/include/page_dmatrix-inl.hpp
	rm -rf ./output/include/page_fmatrix-inl.hpp
	rm -rf ./output/include/random.h
	rm -rf ./output/include/simple_dmatrix-inl.hpp
	rm -rf ./output/include/simple_fmatrix-inl.hpp
	rm -rf ./output/include/thread_buffer.h
	rm -rf ./output/include/thread.h
	rm -rf ./output/include/utils.h
	rm -rf ./output
	rm -rf src/ipac_io.o
	rm -rf src/ipac_ipac.o
	rm -rf src/ipac_ipac_main.o

.PHONY:dist
dist:
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40mdist[0m']"
	tar czvf output.tar.gz output
	@echo "make dist done"

.PHONY:distclean
distclean:clean
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40mdistclean[0m']"
	rm -f output.tar.gz
	@echo "make distclean done"

.PHONY:love
love:
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40mlove[0m']"
	@echo "make love done"

ipac:src/ipac_io.o \
  src/ipac_ipac.o \
  src/ipac_ipac_main.o
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40mipac[0m']"
	$(CXX) src/ipac_io.o \
  src/ipac_ipac.o \
  src/ipac_ipac_main.o -Xlinker "-("   -lpthread -lgomp \
  -lcrypto \
  -lrt -Xlinker "-)" -o ipac
	mkdir -p ./output/bin
	cp -f --link ipac ./output/bin

libipac.a:src/ipac_io.o \
  src/ipac_ipac.o \
  src/ipac_ipac_main.o \
  include/data.h \
  include/dmatrix.h \
  include/io.h \
  include/io_utils.h \
  include/ipac.h \
  include/iterator.h \
  include/matrix_csr.h \
  include/omp_xg.h \
  include/page_dmatrix-inl.hpp \
  include/page_fmatrix-inl.hpp \
  include/random.h \
  include/simple_dmatrix-inl.hpp \
  include/simple_fmatrix-inl.hpp \
  include/thread_buffer.h \
  include/thread.h \
  include/utils.h
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40mlibipac.a[0m']"
	ar crs libipac.a src/ipac_io.o \
  src/ipac_ipac.o \
  src/ipac_ipac_main.o
	mkdir -p ./output/lib
	cp -f --link libipac.a ./output/lib
	mkdir -p ./output/include
	cp -f --link include/data.h include/dmatrix.h include/io.h include/io_utils.h include/ipac.h include/iterator.h include/matrix_csr.h include/omp_xg.h include/page_dmatrix-inl.hpp include/page_fmatrix-inl.hpp include/random.h include/simple_dmatrix-inl.hpp include/simple_fmatrix-inl.hpp include/thread_buffer.h include/thread.h include/utils.h ./output/include

libipac.so:src/ipac_io.o \
  src/ipac_ipac.o \
  src/ipac_ipac_main.o \
  include/data.h \
  include/dmatrix.h \
  include/io.h \
  include/io_utils.h \
  include/ipac.h \
  include/iterator.h \
  include/matrix_csr.h \
  include/omp_xg.h \
  include/page_dmatrix-inl.hpp \
  include/page_fmatrix-inl.hpp \
  include/random.h \
  include/simple_dmatrix-inl.hpp \
  include/simple_fmatrix-inl.hpp \
  include/thread_buffer.h \
  include/thread.h \
  include/utils.h
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40mlibipac.so[0m']"
	$(CXX) -shared src/ipac_io.o \
  src/ipac_ipac.o \
  src/ipac_ipac_main.o -Xlinker "-("  -lpthread -lgomp \
  -lcrypto \
  -lrt -Xlinker "-)" -o libipac.so
	mkdir -p ./output/so
	cp -f --link libipac.so ./output/so
	mkdir -p ./output/include
	cp -f --link include/data.h include/dmatrix.h include/io.h include/io_utils.h include/ipac.h include/iterator.h include/matrix_csr.h include/omp_xg.h include/page_dmatrix-inl.hpp include/page_fmatrix-inl.hpp include/random.h include/simple_dmatrix-inl.hpp include/simple_fmatrix-inl.hpp include/thread_buffer.h include/thread.h include/utils.h ./output/include

src/ipac_io.o:src/io.cpp \
  include/io.h \
  include/data.h \
  include/utils.h \
  include/iterator.h \
  include/dmatrix.h \
  include/io_utils.h \
  include/utils.h \
  include/io_utils.h \
  include/utils.h \
  include/simple_dmatrix-inl.hpp \
  include/io.h \
  include/simple_fmatrix-inl.hpp \
  include/random.h \
  include/omp_xg.h \
  include/matrix_csr.h \
  include/io.h \
  include/omp_xg.h \
  include/page_dmatrix-inl.hpp \
  include/thread_buffer.h \
  include/thread.h \
  include/page_fmatrix-inl.hpp
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40msrc/ipac_io.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) $(CPPFLAGS) $(CXXFLAGS)  -o src/ipac_io.o src/io.cpp

src/ipac_ipac.o:src/ipac.cpp \
  include/ipac.h \
  include/data.h \
  include/utils.h \
  include/iterator.h \
  include/io.h \
  include/dmatrix.h \
  include/io_utils.h \
  include/utils.h
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40msrc/ipac_ipac.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) $(CPPFLAGS) $(CXXFLAGS)  -o src/ipac_ipac.o src/ipac.cpp

src/ipac_ipac_main.o:src/ipac_main.cpp \
  include/ipac.h \
  include/data.h \
  include/utils.h \
  include/iterator.h \
  include/io.h \
  include/dmatrix.h \
  include/io_utils.h \
  include/utils.h
	@echo "[[1;32;40mIPAC:BUILD[0m][Target:'[1;32;40msrc/ipac_ipac_main.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) $(CPPFLAGS) $(CXXFLAGS)  -o src/ipac_ipac_main.o src/ipac_main.cpp

endif #ifeq ($(shell uname -m),x86_64)


