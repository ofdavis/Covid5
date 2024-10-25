clear all 
cd "/users/owen/Covid5"

copy "https://www.ssa.gov/open/data/fy08-present-rib-filed-via-internet.csv" /// 
	data/ssa1.csv, replace
	
copy "https://www.ssa.gov/open/data/fy12-onward-rib-filed-via-internet.csv" ///
	data/ssa2.csv, replace
	
*----------------------- 2008-2010 data -------------------------------
import delimited data/ssa1.csv, clear varnames(1)
drop internetsocialsecurityretirement percentagefiledviatheinternet

destring fiscalyear, ignore("*") gen(year)
replace year = year-1 if inlist(month,"OCTOBER","NOVEMBER","DECEMBER")

drop if month=="TOTAL"
gen yearstr = strofreal(year)
gen date = date(yearstr + month + "1", "YMD")

format date %td
gen mo = ym(year(date),month(date))
format mo %tm

destring totalsocialsecurityretirementins, ignore(",") gen(claims)
drop fiscalyear month yearstr totalsocialsecurityretirementins date

tempfile ssa1
save "`ssa1'"
 

*----------------------- 2008-2010 data -------------------------------
import delimited data/ssa2.csv, clear

rename fiscalyearfielda year 
rename monthfieldb month 
rename totalsocialsecurityretirementins claims 
drop internetsocialsecurityretirement percentagefiledviatheinternetfie commentsfieldf


replace year=year-1 if inlist(month, "Oct","Nov","Dec")
gen yearstr = strofreal(year)
gen date = date(yearstr + month + "1", "YMD")
format date %td
gen mo = ym(year(date),month(date))
format mo %tm

destring claims, ignore(",") replace

keep year claims mo 

append using "`ssa1'"

gen month = month(dofm(mo))
sort mo 


*----------------------- population data CPS -------------------------------
frame2 pop, replace 
use data/covid_data, clear

* gen mo = ym(year,month)
keep mo age wtfinl
keep if inrange(age,62,70)

gen pop  = 1 
*gen pop1 = inrange(age,62,64)
*gen pop2 = inrange(age,65,67)
*gen pop3 = inrange(age,68,70)

local popvar ""
forvalues a=62/70 { 
	gen pop`a'=age==`a'
	local popvar = "`popvar' " + "pop`a'"
}

collapse (sum) pop `popvar' (mean) age [fw=wtfinl], by(mo)
tempfile pop 
save "`pop'"
frame change default 
merge 1:1 mo using "`pop'"
keep if _merge==3
drop _merge


*----------------------- finalize and estimate -------------------------------
tsset mo 
replace claims = claims/1000
tssmooth ma claims_ma = claims, window(11 1)

* predict claims 
cap drop pclaims*
reg claims mo pop62 pop63 pop64 pop65 pop66 pop67 pop68 pop69 pop70 i.month if mo<`=tm(2020m3)'
predict pclaims
tssmooth ma pclaims_ma = pclaims, window(11 1)

colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
tsline claims claims_ma  pclaims_ma /*pclaims*/  if inrange(year(dofm(mo)),2008,2024), ///
	lc("`r(p1)'%25" "`r(p1)'" "`r(p3)'" "`r(p3)'%50") lw(medium medthick medthick) ///
	xtitle("") ytitle("Claims, 000s") tla(, format(%tmCY)) xline(`=tm(2020m3)', lc(gray) lp(dot)) /// 
	legend(order(1 "Claims" 2 "Claims," "smoothed" 3 "Predicted claims," "smoothed")) xsize(7)

graph export output/figs/ssa.pdf, replace







