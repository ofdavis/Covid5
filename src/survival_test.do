clear all 
* Set the total number of individuals and time periods
global N = 1000
global T = 1000

* Create an empty dataset with the total number of observations (individuals x periods)
set obs $N

* Generate a unique identifier for each individual
gen id = _n

* Random starting values for age, sex 
gen age = 50+floor(runiform()*30)
gen sex = floor(runiform()*2)

* expand 
expand $T
bys id: gen t=_n
sort id t
xtset id t

* create sex- and age-by-time vars; time is 500 
gen sext = sex*(t>=500)
gen aget = age*(t>=500)

* model: h(t) = h0(t) * exp(Xb); h0(t)=p*t^(p-1)
* Xb = b0 + bs*sex + bst*sext + ba*age + bat*aget + e 
global p   1.01
global b0  -15
global bs  1
global bst 1 
global ba  0.1
global bat 0 
gen e = rnormal(0,0.1)

gen h0 = $p * t^($p - 1)
gen h  = h0*exp($b0 + $bs*sex + $bst*sext + $ba*age + $bat*aget + e)

* simulate death 
cap drop d 
gen d = runiform()<h if t==1
replace d = max((d[_n-1]==0)*(runiform()<h),d[_n-1]==1) if t>1

*binscatter d t, by(sex)


* ------------------------ survival data frame  ------------------------
frame copy default surv, replace 
frame change surv 
gen td = t*d 
replace td=1001 if td==0

collapse (first) sex age (min) td (max) d, by(id)
gen t0 = 0 
gen t1 = 1000


* set survival data 
stset td, failure(d) id(id) origin(time t0) time0(t0)

* basic stcox 
sts graph, by(sex)
stcox age sex

* split and rerun stcox 
stsplit td2, at(500)
stcox age sex
gen aget = age*(td2==500)
gen sext = sex*(td2==500)

stcox age aget sex sext
