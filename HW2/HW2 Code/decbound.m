function decide = decbound(r1,r2)

if (abs(r1)+abs(r2)) > (abs(r1+1) + abs(r2+1))
    decide = 's1'
elseif (abs(r1)+abs(r2)) > (abs(r1-1) + abs(r2-1))
    decide = 's2'
else
    decide = '?'
end