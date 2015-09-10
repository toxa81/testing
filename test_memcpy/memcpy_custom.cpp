extern "C" void memcpy_custom(double* out, double* in, int length)
{
    for (int i = 0; i < length; i++) out[i] = in[i];
}