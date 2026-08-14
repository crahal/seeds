// cd ""  set your cwd here
use http://www.stata-press.com/data/r7/school.dta, clear
tempfile resultsfile
postfile myresults str200 seed draws rho21 using `resultsfile'
file open seedfile using "../assets/seed_list.txt", read
file read seedfile line
local total_draws = 150 // Set the total number of draws here
local total_seeds = 10000 // Set the total number of seeds here
local iterations = 0

while r(eof) == 0 & `iterations' < `total_seeds' {
    local seed `line'
    forval draw = 2(2)`total_draws' {
        capture quietly mvprobit (private = years logptax loginc) (vote = years logptax loginc), ///
                  seed(`seed') draws(`draw')
        if _rc != 0 {
            display "Error occurred: `r(ghmessage)'"
            continue, break
        }
        local rho21 = e(rho21)
        post myresults ("`seed'") (`draw') (`rho21')
    }
    local iterations = `iterations' + 1
    display "Completed iteration: `iterations' out of `total_seeds' at `c(current_time)'"
    file read seedfile line
}
file close seedfile
postclose myresults
local filename "results_school_total_draws`total_draws'_total_seeds`total_seeds'.csv"
use `resultsfile', clear
export delimited "../data/mvprobit/`filename'", replace
