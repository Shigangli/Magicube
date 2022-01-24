#include <iostream>
#include <limits>

int main ()
{   
    //char a = 127;
    //char b = 0;
    unsigned char m1 = 15;
    unsigned char a = 223;
    unsigned char b = 0;
    int ttt = 255;
    b = (unsigned char)ttt;
    printf("b_char = %d \n", b);	
    //ttt = 256;
    ttt = 1;
    b = (unsigned char)ttt;
    printf("b_char1 = %d \n", b);	

    int array_a[6] = {1, 2, 3, 4, 5, 6};
    printf("array_a[%d] = %d \n", b, array_a[b]);

    if(*(array_a+3) == 4)
        printf("pointer test correct\n");
    //b = a + 8;
    //printf("b = %d %o %x\n", b, b, b);	
    printf("a = %d %o %x\n", a & (m1 << 4), a & (m1 << 4), a & (m1 << 4));	
    printf("a shift = %d %o %x\n", (a & (m1 << 4)) >> 4, (a & (m1 << 4)) >> 4, (a & (m1 << 4)) >> 4);
    unsigned int aaa = 0xF0000F00;    
    unsigned int bbb = aaa >> 4;    
    printf("aaa = %x, bbb = %x \n", (aaa << 4), (aaa >> 4));	
    printf("a shift = %d %o %x\n", (a & (m1 << 4)) >> 4, (a & (m1 << 4)) >> 4, (a & (m1 << 4)) >> 4);	
    int an = a;
    int n = b;
    int inf_int = 0x7F800000;
    int test_int1 = 0x80000001;
    int mask = 0x0000000F;
    int test_int2 = test_int1 & mask;
    int int_array[2];
    int int_array_1[2];
    int_array[0] = 0x01100110;
    int_array[1] = 0x01111111;
    char * int_array_char = reinterpret_cast<char *>(int_array);
    char * int_array_1_char = reinterpret_cast<char *>(int_array_1);
    for(int i=0; i<2; i++)
        for(int j=0; j<4; j++){
            printf("%x\n", int_array_char[i*4+j]);	
            int_array_1_char[j*2+i] = int_array_char[i*4+j];	
	}
    printf("%x\n", int_array_1[0]);	
    printf("%x\n", int_array_1[1]);	
    
    //printf("%d %d %o\n", test_int1, test_int2, test_int2);
    //printf("%x %o\n", fl, fl);
    ////float inf = (float)0x7F800000;
    //float inf = *(float *)&inf_int;
    ////float inf = 1000000000000000000.0;
    ////float inf = std::numeric_limits<float>::infinity();
    //int fl = *(int*)&inf;
    //printf("%x %o\n", fl, fl);
    //float r = 0.0;
    ////r = inf/2.0;
    //r = inf - 11111111111000.0;
    //std::cout << "inf: " << inf << "inf_over_2: " << r << std::endl;
    //
    //std::cout << "an: " << an << "n: " << n << std::endl;
    //std::cout << "minimum value of int: "
    //          << std::numeric_limits<int>::min()
    //          << std::endl;
}
