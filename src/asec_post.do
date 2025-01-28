use data/generated/asec_data.dta, clear 
rename statefip state
merge m:1 state county using data/generated/housing
merge 1:1 asecidp mo using data/generated/pred_asec_R, gen(_merge_p)

reghdfe retired i.covid##c.p_retired age agesq agecub [pw=asecwt], absorb(educ race sex married nativity diffrem diffphys diffmob )


* --------------------------- collapse ----------------------------------------* 
gen pop=1
collapse (mean) retired p_retired (sum) pop (first) dp [fw=asecwt], by(year state county own)
gen diff = retired-p_retired

twoway (scatter diff dp if year>=2021 & own==1 [w=pop]) (lfit diff dp if year>=2021 & own==1 [w=pop])
reghdfe diff dp if year>=2021 & own==1 [pw=pop], abs(year)
reg diff incrd if year==2024 [pw=pop]
