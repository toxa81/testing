#include <memory>
#include <iostream>

template <typename T>
class A
{
    private:
        std::unique_ptr<T[]> unique_ptr_;
    public:
        A()
        {
        }
        A(A const& src) = delete;
        A& operator=(A const& src) = delete;
        A(A&& src) : unique_ptr_(std::move(src.unique_ptr_))
        {
        }
        A& operator=(A&& src)
        {
            std::cout << "in A::A(A&& src)" << std::endl;
            unique_ptr_ = std::move(src.unique_ptr_);
            return *this;
        }
};

template <typename T>
class B
{
    private:
        A<T> a_;
    public:
        B()
        {
        }
        
        //B(B const& src) = delete;
        //B& operator=(B const& src) = delete;
        //B(B&& src)
        //{

        //}
        //B& operator=(B&& src)
        //{
        //    std::cout << "in B::B(B&& src)" << std::endl;
        //    return *this;
        //}
};

int main(int argn, char** argv)
{
    A<double> a1;
    A<double> a2;

    a1 = std::move(a2);
    B<double> b0;
    B<double> b1[2];
    
    b1[1] = std::move(b0);
    b1[0] = B<double>();
}
