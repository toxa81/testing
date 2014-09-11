program proc_status
implicit none
character(1024) line, str1
integer idx, vmhwm,vmrss
integer, external :: getpid_

write(str1,*)getpid()
line="/proc/"//trim(adjustl(str1))//"/status"
write(*,*)line

open(100,file=trim(adjustl(line)),form="formatted")
do while(.true.)
  read(100,'(A)',end=10)line
  idx = index(line, "VmHWM:")
  if (idx.ne.0) then
    str1 = trim(adjustl(line(7:1024)))
    idx = index(str1, " ")
    read(str1(1:idx),*)vmhwm
    if (str1(idx+1:idx+3).ne."kB") stop("wrong units")
    vmhwm = vmhwm * 1024
  endif
  idx = index(line, "VmRSS:")
  if (idx.ne.0) then
    str1 = trim(adjustl(line(7:1024)))
    idx = index(str1, " ")
    read(str1(1:idx),*)vmrss
    if (str1(idx+1:idx+3).ne."kB") stop("wrong units")
    vmrss = vmrss * 1024
  endif
end do
10 continue
close(100)

write(*,'("Peak resident set size (high water mark) : ",I10, " bytes")')vmhwm
write(*,'("Resident set size                        : ",I10, " bytes")')vmrss

return
end program
