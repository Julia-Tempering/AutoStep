using FFTW, MCMCChains, CSV, DataFrames, Distributions
include("utils.jl")

function main()
	new_df = DataFrame(explorer = [], model = [], seed = [],
		mean_1st_dim = [], var_1st_dim = [], time = [], n_logprob = [], n_steps = [], acceptance_prob = [], energy_jump_dist = [],
		miness = [], minKSess = [], geyerESS = [])
	for model in ["mixture", "funnel2", "funnel100", "kilpisjarvi", "mRNA", "orbital"]
		reference_samples = CSV.read("icml2025/samples/$(model).csv", DataFrame)
		df = CSV.read("icml2025/exp_results_$(model)_temp1.csv", DataFrame)
		model_dim = model == "funnel2" ? 2 :
            model == "funnel100" ? 100 :
            model == "kilpisjarvi" ? 3 :
            model == "mRNA" ? 5 :
            model == "orbital" ? 12 :
			model == "mixture" ? 500 :
            error("Unknown model: $model")
		for explorer in ["adaptive RWMH", "adaptive MALA", "AutoStep RWMH", "AutoStep MALA", "DRHMC", "HitAndRunSlicer", "NUTS"]
			alg = if explorer == "adaptive RWMH"
				"adaptive_rwmh"
			elseif explorer == "adaptive MALA"
				"adaptive_mala"
			elseif explorer == "AutoStep RWMH (precond)"
				"adaptive_rwmh_precond"
			elseif explorer == "AutoStep MALA (precond)"
				"adaptive_mala_precond"
			elseif explorer == "NUTS"
				"nuts"
			elseif explorer == "AutoStep MALA"
				"automala"
			elseif explorer == "AutoStep RWMH"
				"autorwmh"
			elseif explorer == "DRHMC"
				"drhmc"
			else
				"slicer"
			end
			for seed in 1:10
				samples = CSV.read("icml2025/temp/$(seed)_$(model)_$(alg).csv", DataFrame)
				# read experiment results from saved files
				row_index = findfirst(eachrow(df)) do row
					row.model == model &&
						row.seed == seed &&
						row.explorer == explorer
				end
				print(row_index)
				time = df.time[row_index]
				n_logprob = df.n_logprob[row_index]
				n_steps = df.n_steps[row_index]
				acceptance_prob = df.acceptance_prob[row_index]
				energy_jump_dist = df.energy_jump_dist[row_index]
				miness = df.miness[row_index]
				# compute new results
				samples_1st_dim = collect(samples[1, :])
				mean_1st_dim = mean(samples_1st_dim)
				var_1st_dim = var(samples_1st_dim)
				minKSess = KSess_two_sample(samples_1st_dim, collect(reference_samples[1, :]))
				geyeress = ess(samples_1st_dim; autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))
				for j in 2:model_dim
					xs = collect(samples[j, :])
					if model == "funnel100" || model == "mixture"
						minKSess = min(minKSess, KSess_two_sample(xs, collect(reference_samples[2, :])))
					else
						minKSess = min(minKSess, KSess_two_sample(xs, collect(reference_samples[j, :])))
					end
					geyeress = min(geyeress, ess(xs; autocov_method = FFTAutocovMethod(), maxlag = typemax(Int)))
				end
				# print and save
				println("$model, $explorer, $seed, $geyeress")
				push!(new_df, [explorer, model, seed, mean_1st_dim, var_1st_dim, time, n_logprob, n_steps, acceptance_prob,
					energy_jump_dist, miness, minKSess, geyeress])
			end
		end
	end
	CSV.write("icml2025/results/exp_results_$(model).csv", new_df)
end

main()