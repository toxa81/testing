program test_libm
implicit none
integer n
real(8), external :: omp_get_wtime
real(8) :: t
complex(8) :: z
integer j
!
n = 10000000
z = dcmplx(0.d0, 0.d0)
!
t = -omp_get_wtime()
do j = 1, 10
  call run_test(n, z)
enddo
t = t + omp_get_wtime()
!
write(*,*)'time=',t,' sec.'
write(*,*)'z=',z

end program

subroutine run_test(n, z)
implicit none
integer, intent(in) :: n
complex(8), intent(out) :: z
integer :: i
real(8) d,p

d = 1.d0 / n
do i = 1, n
  p = d * i * 2 * 3.141592d0
  z = z + exp(dcmplx(0.d0, p))
enddo
end subroutine
