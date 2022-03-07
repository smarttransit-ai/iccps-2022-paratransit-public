using Distributed
using SharedArrays

addprocs(25)
#addprocs(1)

@everywhere begin
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using DataFrames
using CSV
using Combinatorics
using StatsBase
using Random
using Dates
using BenchmarkTools
using MCTS
using D3Trees
using TickTock
using CPUTime
#using Profile
#using ProfileVega
#using StatProfilerHTML
#using Base: get

ENV["COLUMNS"]=100
cd("..")
pwd()
end

@everywhere begin
function LoadChains(file_path::String, request_arrive::Int64, window::Int64)
    # request_arrive is in minutes before requested pickup (ie 60)
    # window is minutes (ie 15)
    chains = DataFrame(CSV.File(file_path))
    chains = transform(chains, :pickup_node_id => ByRow(x -> x + 1) => :pickup_node_id)
    chains = transform(chains, :dropoff_node_id => ByRow(x -> x + 1) => :dropoff_node_id)
    chains = transform(chains, :chain_id => ByRow(x -> x + 1) => :chain_id)
    chains = transform(chains, :chain_order => ByRow(x -> x + 1) => :chain_order)
    chains = transform(chains, :pickup_time_since_midnight => ByRow(x -> x - request_arrive * 60) => :request_arrive_time)
    chains = transform(chains, :pickup_time_since_midnight => ByRow(x -> x - window * 60) => :e)
    chains = transform(chains, :dropoff_time_since_midnight => ByRow(x -> x + window * 60) => :l)
    return chains
end

function GetChain(df_chain, chain_id)
    df = filter(row -> row.chain_id == chain_id, df_chain)
    df = sort(df, :chain_order)
    result = Vector{Request}(undef, 0)
    for row in eachrow(df)
        push!(result, Request(row.chain_order, row.pickup_node_id, row.dropoff_node_id, row.request_arrive_time, row.e, row.l, -1, -1, -1, chain_id))
    end
    result = filter(x -> x.pickup_node_id != x.dropoff_node_id, result)
    for i in 1:length(result)
        result[i].request_id = i
    end
    return result
end

function PreprocessChains(df_chain)
    chain_ids = sort(unique(df_chain.chain_id))
    my_chains = [GetChain(df_chain, chain_id) for chain_id in chain_ids]
    return my_chains
end
end

# load travel_time_matrix
@everywhere begin
file_path = "data/travel_time_matrix/travel_time_matrix.csv"
#file_path = "data/travel_time_matrix/travel_time_matrix_cong_mike.csv"
TRAVEL_TIME_MATRIX = DataFrame(CSV.File(file_path, header=0))
TRAVEL_TIME_MATRIX = Matrix(TRAVEL_TIME_MATRIX)
TRAVEL_TIME_MATRIX = map(round, TRAVEL_TIME_MATRIX)
TRAVEL_TIME_MATRIX = map(Int32, TRAVEL_TIME_MATRIX)

# load nodes
file_path = "data/travel_time_matrix/nodes.csv"
nodes = DataFrame(CSV.File(file_path))

# load chains
DF_TRAIN = LoadChains("data/CARTA/processed/train_chains.csv", 60, 15)
DF_TEST = LoadChains("data/CARTA/processed/test_chains.csv", 60, 15)
#DF_TRAIN = LoadChains("data/CARTA/processed/train_chains_cong.csv", 60, 15)
#DF_TEST = LoadChains("data/CARTA/processed/test_chains_cong.csv", 60, 15)
end

println("loaded data")
flush(stdout)

@everywhere begin
struct theta
    a::Int64 # planned arrival time to theta_ki (seconds past midnight)
    e::Int64 # if pickup location then set to early time window e_n, else -1
    l::Int64 # if dropoff location then set to late time window l_n, else -1
    p::Int64 # 0 if pickup location, 1 if dropoff location
    node::Int64 # node id
    request_id::Int64
    chain_id::Int64
end

mutable struct Request
    request_id::Int64
    pickup_node_id::Int64
    dropoff_node_id::Int64
    request_arrive_time::Int64
    e::Int64
    l::Int64
    actual_pickup_time::Int64
    actual_dropoff_time::Int64
    vehicle_id::Int64
    chain_id::Int64
end

struct CentMDPState
    time::Int64
    requests::Vector{Request}
    request_id::Int64
    vehicle_locations::Vector{theta}
    route_plans::Vector{Vector{theta}}
    done::Bool
end

struct CentAction
    route_plans::Vector{Vector{theta}}
    utility::Float64
end

struct RVEdge
    vehicle_id::Int64
    route_plan::Vector{theta}
    utility::Float64
end

struct VVEdge
    v1::Int64
    v2::Int64
    v1_route_plan::Vector{theta}
    v2_route_plan::Vector{theta}
    utility::Float64
end

struct CentMDP <: MDP{CentMDPState, CentAction} # Note that our MDP is parametarized by the state and the action
    num_vehicles::Int64
    capacity::Int64
    k_max::Int64 # max number of feasible actions to return
    vv::Bool
    budget::Bool
end
end

@everywhere begin
const TEST_CHAINS = PreprocessChains(DF_TEST)
const TRAIN_CHAINS = PreprocessChains(DF_TRAIN)
end

@everywhere begin
function thetaEqual(c1::theta, c2::theta)
    if (c1.p == c2.p) && (c1.request_id == c2.request_id) && (c1.node == c2.node) && (c1.chain_id == c2.chain_id)
        return true
    else
        return false
    end
end

function RouteEqual(r1::Vector{theta}, r2::Vector{theta})
    if length(r1) != length(r2)
        return false
    else
        for i in 1:length(r1)
            if !thetaEqual(r1[i], r2[i])
                return false
            end
        end
    end
    return true
end

function RoutePlansEquals(r1::Vector{Vector{theta}}, r2::Vector{Vector{theta}})
    for v in 1:length(r1)
        c1 = r1[v]
        c2 = r2[v]
        if !RouteEqual(c1, c2)
            return false
        end
    end
    return true
end

function VehicleLocationsEquals(c1::Vector{theta}, c2::Vector{theta})
    return RouteEqual(c1, c2)
end

function RequestEquals(r1::Request, r2::Request)
    if (r1.request_id == r2.request_id) && (r1.pickup_node_id == r2.pickup_node_id) && (r1.dropoff_node_id == r2.dropoff_node_id) && (r1.request_arrive_time == r2.request_arrive_time) && (r1.chain_id == r2.chain_id)
        return true
    else
        return false
    end
end

function CentActionEqual(a1::CentAction, a2::CentAction)
    return RoutePlansEquals(a1.route_plans, a2.route_plans)
end

function CentMDPStateEqual(s1::CentMDPState, s2::CentMDPState)
    if (s1.time == s2.time) && (s1.request_id == s2.request_id) && VehicleLocationsEquals(s1.vehicle_locations, s2.vehicle_locations) && RoutePlansEquals(s1.route_plans, s2.route_plans) && (s1.done == s2.done) && RequestEquals(s1.requests[s1.request_id], s2.requests[s2.request_id])
        return true
    else
        return false
    end
end
end

@everywhere begin
using Base

Base.:(==)(s1::CentMDPState, s2::CentMDPState) = CentMDPStateEqual(s1, s2)

function Base.get(state_map::Dict{CentMDPState,Int}, s::CentMDPState, d)
    for k in keys(state_map)
        if k == s
            return state_map[k]
        end
    end
    return d
end
end

@everywhere begin
function CheckCapacity(m::CentMDP, route_plan::Vector{theta})
    num_passengers = length(filter(x -> x.p == 1, route_plan)) - length(filter(x -> x.p == 0, route_plan))
    if num_passengers > m.capacity
        return false
    end
    for loc in route_plan
        if loc.p == 0
            num_passengers = num_passengers + 1
        else
            num_passengers = num_passengers - 1
        end
        if num_passengers > m.capacity
            return false
        end
    end
    return true
end

function CheckFeasibleRoutePlan(s::CentMDPState, m::CentMDP, new_route_plan::Vector{theta}, vehicle_id::Int64)
    #route_plan = deepcopy(new_route_plan)
    route_plan = new_route_plan
    
    # check passenger capacity
    if !CheckCapacity(m, route_plan)
        return false
    end
    
    # update a(thetas), ensure constraints are satisfied
    time_to_location = s.vehicle_locations[vehicle_id].a
    if time_to_location < s.time
        time_to_location = s.time
    end
    last_node = s.vehicle_locations[vehicle_id].node
    for k in 1:length(route_plan)
        min_time_to_location = time_to_location + TRAVEL_TIME_MATRIX[last_node, route_plan[k].node]
        if route_plan[k].p == 0
            if min_time_to_location < route_plan[k].e
                time_to_location = route_plan[k].e
            else
                time_to_location = min_time_to_location
            end
        else
            time_to_location = min_time_to_location
        end
        route_plan[k] = theta(time_to_location, route_plan[k].e, route_plan[k].l, route_plan[k].p, route_plan[k].node, route_plan[k].request_id, route_plan[k].chain_id)
        last_node = route_plan[k].node
            
        if (route_plan[k].p == 1) && (route_plan[k].a > route_plan[k].l)
            return false
        elseif (route_plan[k].p == 0) && (route_plan[k].a < route_plan[k].e)
            return false
        elseif (route_plan[k].p == 0) && (length(filter(x -> x.request_id == route_plan[k].request_id, route_plan[k+1:length(route_plan)])) != 1)
            return false
        end
    end
    return route_plan
end

function TSPInsertion(s::CentMDPState, m::CentMDP, current_route_plan::Vector{theta}, vehicle_id::Int64, request::Request)
    # initialize pickup and dropoff thetas
    pickup_theta = theta(-1, request.e, -1, 0, request.pickup_node_id, request.request_id, request.chain_id)
    dropoff_theta = theta(-1, -1, request.l, 1, request.dropoff_node_id, request.request_id, request.chain_id)
    
    # insert pickup and dropoff, check if resulting route plans are feasible
    route_plans = Vector{Vector{theta}}(undef, 0)
    for i in 1:length(current_route_plan)+1
        insert_pickup = vcat(current_route_plan[1:i-1], pickup_theta, current_route_plan[i:length(current_route_plan)])
        for j in i+1:length(current_route_plan)+2
            new_route_plan = vcat(insert_pickup[1:j-1], dropoff_theta, insert_pickup[j:length(insert_pickup)])
            feasible_route_plan = CheckFeasibleRoutePlan(s, m, new_route_plan, vehicle_id)
            if feasible_route_plan != false
                push!(route_plans, feasible_route_plan)
            end
        end
    end
    return route_plans
end
end

@everywhere begin
function RoutePlanCost(m::CentMDP, s::CentMDPState, route_plan::Vector{theta}, vehicle_id::Int64)
    result = 0.0
    dropoff_nodes = filter(x -> x.p == 1, route_plan)
    for dropoff_node in dropoff_nodes
        pickup_nodes = filter(x -> (x.p == 0) && (x.request_id == dropoff_node.request_id), route_plan)
        if length(pickup_nodes) != 0
            result = result + dropoff_node.a - pickup_nodes[1].a
        else
            if s.vehicle_locations[vehicle_id].request_id == dropoff_node.request_id
                ti = s.vehicle_locations[vehicle_id].a
            else
                ti = s.time
            end
            result = result + dropoff_node.a - ti
        end
    end
    return result
end
    
function RoutePlanCostBudgetVoid(m::CentMDP, s::CentMDPState, route_plan::Vector{theta}, vehicle_id::Int64)
    #println("start RoutePlanCostBudget")
    #println("route_plan: $(route_plan)")
    result = m.capacity * (s.requests[length(s.requests)].l - s.time)
    num_passengers = length(filter(x -> x.p == 1, route_plan)) - length(filter(x -> x.p == 0, route_plan))
    current_time = s.time
    #println("result: $(result), num_passengers: $(num_passengers), current_time: $(current_time)")
    for loc in route_plan
        result = result - (num_passengers * (loc.a - current_time))
        if loc.p == 0
            num_passengers = num_passengers + 1
        else
            num_passengers = num_passengers - 1
        end
        current_time = loc.a
        #println("result: $(result), num_passengers: $(num_passengers), current_time: $(current_time)")
    end
    #println("end ROutePlanCostBudget")
    return result
end
    
function RoutePlanCostBudget(m::CentMDP, s::CentMDPState, route_plan::Vector{theta}, vehicle_id::Int64)
    result = m.capacity * (s.requests[length(s.requests)].l - s.time)
    num_passengers = length(filter(x -> x.p == 1, route_plan)) - length(filter(x -> x.p == 0, route_plan))
    current_time = s.time
    for loc in route_plan
        if num_passengers > 0
            result = result - (loc.a - current_time)
        end
        if loc.p == 0
            num_passengers = num_passengers + 1
        else
            num_passengers = num_passengers - 1
        end
        current_time = loc.a
    end
    return result
end

function GetRVEdge(m::CentMDP, s::CentMDPState, v::Int64)
    #current_request = deepcopy(s.requests[s.request_id])
    current_request = s.requests[s.request_id]
    current_route_plan = s.route_plans[v]
    if m.budget
        c_old = RoutePlanCostBudget(m, s, s.route_plans[v], v)
    else
        c_old = RoutePlanCost(m, s, s.route_plans[v], v)
    end
    new_route_plans = TSPInsertion(s, m, current_route_plan, v, current_request)
    new_rv_edges = Vector{RVEdge}(undef, 0)
    for new_route_plan in new_route_plans
        if m.budget
            c_new = RoutePlanCostBudget(m, s, new_route_plan, v)
            weight = c_new - c_old
        else
            c_new = RoutePlanCost(m, s, new_route_plan, v)
            weight = c_new - c_old
        end
        push!(new_rv_edges, RVEdge(v, new_route_plan, weight))
    end
    return new_rv_edges
end

function GetVVEdgeDirected(m::CentMDP, s::CentMDPState, v1::Int64, v2::Int64)
    if m.budget
        c_v1_old = RoutePlanCostBudget(m, s, s.route_plans[v1], v1)
        c_v2_old = RoutePlanCostBudget(m, s, s.route_plans[v2], v2)
    else
        c_v1_old = RoutePlanCost(m, s, s.route_plans[v1], v1)
        c_v2_old = RoutePlanCost(m, s, s.route_plans[v2], v2)
    end
    feasible_actions = Vector{VVEdge}(undef, 0)
    pickups = filter(x -> x.p==0, s.route_plans[v1])
    v1_swap_requests = [s.requests[x.request_id] for x in pickups]
    for request in v1_swap_requests
        v1_route_plan = filter(x -> x.request_id != request.request_id, s.route_plans[v1])
        v1_route_plan = CheckFeasibleRoutePlan(s, m, v1_route_plan, v1)
        if v1_route_plan != false
            if m.budget
                c_v1_new = RoutePlanCostBudget(m, s, v1_route_plan, v1)
            else
                c_v1_new = RoutePlanCost(m, s, v1_route_plan, v1)
            end
            v2_route_plans = TSPInsertion(s, m, s.route_plans[v2], v2, request)
            for v2_route_plan in v2_route_plans
                if m.budget
                    c_v2_new = RoutePlanCostBudget(m, s, v2_route_plan, v2)
                    weight = c_v1_new + c_v2_new - c_v1_old - c_v2_old - .1
                else
                    c_v2_new = RoutePlanCost(m, s, v2_route_plan, v2)
                    weight = c_v1_new + c_v2_new - c_v1_old - c_v2_old + .1
                end
                push!(feasible_actions, VVEdge(v1, v2, v1_route_plan, v2_route_plan, weight))
            end
        end
    end
    return feasible_actions
end

function GetVVEdge(m::CentMDP, s::CentMDPState, v1::Int64, v2::Int64)
    v1_feasible_actions = GetVVEdgeDirected(m, s, v1, v2)
    v2_feasible_actions = GetVVEdgeDirected(m, s, v2, v1)
    feasible_actions = vcat(v1_feasible_actions, v2_feasible_actions)
    if length(feasible_actions) > 0
        if m.budget
            best_vv_index = sortperm(map(x -> x.utility, feasible_actions), rev=true)[1]
        else
            best_vv_index = sortperm(map(x -> x.utility, feasible_actions), rev=false)[1]
        end
        return feasible_actions[best_vv_index]
    else
        return false
    end
end

function GetRVEdges(m::CentMDP, s::CentMDPState)
    rv_edges = Vector{RVEdge}(undef, 0)
    for v in 1:m.num_vehicles
        new_rv_edges = GetRVEdge(m, s, v)
        for new_rv_edge in new_rv_edges
            push!(rv_edges, new_rv_edge)
        end
    end
    return rv_edges
end

function GetVVEdges(m::CentMDP, s::CentMDPState)
    vv_edges = Vector{VVEdge}(undef, 0)
    vehicle_combos = collect(combinations(1:m.num_vehicles,2))
    for vehicle_combo in vehicle_combos
        new_vv_edge = GetVVEdge(m, s, vehicle_combo[1], vehicle_combo[2])
        if new_vv_edge != false
            push!(vv_edges, new_vv_edge)
        end
    end
    return vv_edges
end
end

@everywhere begin
function RVOnlyActions(m::CentMDP, s::CentMDPState)
    feasible_actions = Vector{CentAction}(undef, 0)
    rv_edges = GetRVEdges(m, s)
    for rv_edge in rv_edges
        route_plans = deepcopy(s.route_plans)
        route_plans[rv_edge.vehicle_id] = rv_edge.route_plan
        push!(feasible_actions, CentAction(route_plans, rv_edge.utility))
    end
    return feasible_actions
end

function RVandVVActions(m::CentMDP, s::CentMDPState)
    feasible_actions = Vector{CentAction}(undef, 0)
    rv_edges = GetRVEdges(m, s)
    vv_edges = GetVVEdges(m, s)
    for rv_edge in rv_edges
        # RV action
        total_utility = rv_edge.utility
        new_route_plans = deepcopy(s.route_plans)
        new_route_plans[rv_edge.vehicle_id] = rv_edge.route_plan
        push!(feasible_actions, CentAction(new_route_plans, total_utility))
        
        # RV and VV actions
        possible_swaps = deepcopy(vv_edges)
        possible_swaps = filter(x -> (x.v1 != rv_edge.vehicle_id) && (x.v2 != rv_edge.vehicle_id), possible_swaps)
        while length(possible_swaps) > 0
            if m.budget
                new_swap_index = sortperm(map(x -> x.utility, possible_swaps), rev=true)[1]
            else
                new_swap_index = sortperm(map(x -> x.utility, possible_swaps), rev=false)[1]
            end
            new_swap = possible_swaps[new_swap_index]
            new_route_plans[new_swap.v1] = new_swap.v1_route_plan
            new_route_plans[new_swap.v2] = new_swap.v2_route_plan
            total_utility += new_swap.utility
            push!(feasible_actions, CentAction(new_route_plans, total_utility))
            possible_swaps = filter(x -> (x.v1 != new_swap.v1) && (x.v1 != new_swap.v2) && (x.v2 != new_swap.v1) && (x.v2 != new_swap.v2), possible_swaps)
        end
    end
    return feasible_actions
end

function POMDPs.actions(m::CentMDP, s::CentMDPState)
    if m.vv
        feasible_actions = RVandVVActions(m, s)
    else
        feasible_actions = RVOnlyActions(m, s)
    end
    
    if length(feasible_actions) == 0
        push!(feasible_actions, CentAction(s.route_plans, 0.0))
    else
        if m.budget
            feasible_actions = feasible_actions[sortperm(map(x -> x.utility, feasible_actions), rev=true)]
        else
            feasible_actions = feasible_actions[sortperm(map(x -> x.utility, feasible_actions), rev=false)]
        end
        if length(feasible_actions) > m.k_max
            feasible_actions = feasible_actions[1:m.k_max]
        end
    end
    #println("Request ID: $(s.request_id), Action utilities: $(map(x->x.utility, feasible_actions))")
    return feasible_actions
end
end

# random solver

struct MyRandomSolver <: Solver
    seed::Int
end

struct MyRandomPlanner{M} <: Policy
    m::M
    seed::Int
end

POMDPs.solve(sol::MyRandomSolver, m) = MyRandomPlanner(m, sol.seed)

function POMDPs.action(p::MyRandomPlanner{<:MDP}, s)
    feasible_actions = actions(p.m, s)
    act = rand(feasible_actions)
    return act
end

# myopic solver
@everywhere begin
struct MyMyopicSolver <: Solver
    seed::Int
end

struct MyMyopicPlanner{M} <: Policy
    m::M
    seed::Int
end

POMDPs.solve(sol::MyMyopicSolver, m) = MyMyopicPlanner(m, sol.seed)

function POMDPs.action(p::MyMyopicPlanner{<:MDP}, s::CentMDPState)
    feasible_actions = actions(p.m, s)
    act = feasible_actions[1]
    return act
end
end

# MCTS fast rollout
@everywhere begin
struct MCTSRolloutSolver <: Solver
    seed::Int
end

struct MCTSRolloutPlanner{M} <: Policy
    m::M
    seed::Int
end

POMDPs.solve(sol::MCTSRolloutSolver, m) = MCTSRolloutPlanner(m, sol.seed)

function POMDPs.action(p::MCTSRolloutPlanner{<:MDP}, s::CentMDPState)
    feasible_actions = RVOnlyActions(p.m, s)
    if length(feasible_actions) == 0
        act = CentAction(s.route_plans, 0.0)
    else
        if p.m.budget
            feasible_actions = feasible_actions[sortperm(map(x -> x.utility, feasible_actions), rev=true)]
        else
            feasible_actions = feasible_actions[sortperm(map(x -> x.utility, feasible_actions), rev=false)]
        end
        act = feasible_actions[1]
    end
    return act
end
end

# MCTS
@everywhere begin
struct RootMCTSSolver <: Solver
    n_iterations::Int64
    depth::Int64
    exploration_constant::Float64
    n_chains::Int64
    root_parallel::Bool
    rollout::String # greedy_insertion or greedy_swap
    max_time::Float64
end

struct RootMCTSPlanner{M} <: Policy
    m::M
    n_iterations::Int64
    depth::Int64
    exploration_constant::Float64
    n_chains::Int64
    root_parallel::Bool
    rollout::String # fast 
    max_time::Float64
end

POMDPs.solve(sol::RootMCTSSolver, m) = RootMCTSPlanner(m, sol.n_iterations, sol.depth, sol.exploration_constant, sol.n_chains, sol.root_parallel, sol.rollout, sol.max_time)

function POMDPs.action(p::RootMCTSPlanner{<:MDP}, s::CentMDPState)
    action_map_list = [Dict{CentAction, Float64}() for x in 1:p.n_chains]
    chain_ids = sample(1:length(TRAIN_CHAINS), p.n_chains, replace=false)
    if p.root_parallel
        #Threads.@threads for i = 1:p.n_chains
        #    action_map_list[i] = run_mcts(p, s)
        #end
        action_map_list = pmap((ptemp, stemp, chain_id)->run_mcts(ptemp, stemp, chain_id), [p for x in 1:p.n_chains], [s for x in 1:p.n_chains], chain_ids)
    else
        for i in 1:p.n_chains
            action_map_list[i] = run_mcts(p, s, chain_ids[i])
        end
    end
    action = SelectAction(s, action_map_list)
    return action
end

function SelectAction(s::CentMDPState, action_map_list::Vector{Dict{CentAction, Float64}})
    feasible_actions = Vector{CentAction}(undef, 0)
    action_scores = Vector{Vector{Float64}}(undef, 0)

    for action_map in action_map_list
        for key in keys(action_map)
            seen = false
            for j in 1:length(feasible_actions)
                #if key == feasible_actions[j]
                if CentActionEqual(key, feasible_actions[j])
                    push!(action_scores[j], action_map[key])
                    seen = true
                end
            end
            if !seen
                push!(feasible_actions, key)
                push!(action_scores, [action_map[key]])
            end
        end
    end
    scores = map(x -> mean(x), action_scores)
    if length(feasible_actions) > 0
        best_action = feasible_actions[sortperm(scores, rev=true)][1]
    else
        best_action = CentAction(s.route_plans, 0.0)
    end
    return best_action
end

function chain_merge(s::CentMDPState, chain_id::Int64)
    #test_chain = deepcopy(rand(TRAIN_CHAINS))
    #test_chain = rand(TRAIN_CHAINS)
    test_chain = TRAIN_CHAINS[chain_id]
    right_chain = filter(x -> x.request_arrive_time >= s.time, test_chain)
    left_chain = filter(x -> x.request_id <= s.request_id, s.requests)
    if length(right_chain) >= 1
        merged_chain = vcat(left_chain, right_chain)
        result = Vector{Request}(undef, 0)
        for i in 1:length(merged_chain)
            req = merged_chain[i]
            push!(result, Request(i, req.pickup_node_id, req.dropoff_node_id, req.request_arrive_time, req.e, req.l, req.actual_pickup_time, req.actual_dropoff_time, req.vehicle_id, req.chain_id))
            #merged_chain[i].request_id = i
        end
        sr = CentMDPState(s.time, result, length(left_chain), s.vehicle_locations, s.route_plans, false)
    else
        sr = CentMDPState(s.time, s.requests, length(left_chain), s.vehicle_locations, s.route_plans, true)
    end
    return sr
end


function run_mcts(p::RootMCTSPlanner, s::CentMDPState, chain_id::Int64)
    result = Dict{CentAction, Float64}()
    if p.rollout == "fast"
        solver = MCTSSolver(n_iterations=p.n_iterations, depth=p.depth, exploration_constant=p.exploration_constant, max_time=p.max_time, estimate_value=RolloutEstimator(MCTSRolloutSolver(100)))
    else
        solver = MCTSSolver(n_iterations=p.n_iterations, depth=p.depth, exploration_constant=p.exploration_constant, max_time=p.max_time, estimate_value=RolloutEstimator(MyMyopicSolver(100)))
    end
    policy = solve(solver, p.m)

    s2 = chain_merge(s, chain_id)
    if s2.done
        return result
    else
        a = action(policy, s2)
        tree = policy.tree
        sn = get_state_node(tree, s2)
        for san in MCTS.children(sn)
            result[action(san)] = MCTS.q(san)
        end
    end
    return result
end
end

@everywhere begin
function MCTS.build_tree(planner::AbstractMCTSPlanner, s)
    n_iterations = planner.solver.n_iterations
    depth = planner.solver.depth

    if planner.solver.reuse_tree
        tree = planner.tree
    else
        tree = MCTS.MCTSTree{statetype(planner.mdp), actiontype(planner.mdp)}(n_iterations)
    end

    sid = get(tree.state_map, s, 0)
    if sid == 0
        root = MCTS.insert_node!(tree, planner, s)
    else
        root = MCTS.StateNode(tree, sid)
    end

    #start_us = CPUtime_us()
    start_us = time()
    # build the tree
    for n = 1:n_iterations
        MCTS.simulate(planner, root, depth)
        if time() - start_us >= planner.solver.max_time 
            #println(n)
            #flush(stdout)
            break
        end
    end
    return tree
end
end

@everywhere begin
function POMDPs.transition(m::CentMDP, s::CentMDPState, a::CentAction)
    new_request_id = s.request_id + 1 # move to next request index
    # see if we should terminate
    if new_request_id > length(s.requests)
        sp = CentMDPState(s.time, s.requests, s.request_id, s.vehicle_locations, s.route_plans, true)
        return Deterministic(sp)
    end
    
    # get next request and time
    new_request = s.requests[new_request_id] # get next request
    new_time = new_request.request_arrive_time # get next time
    
    #new_route_plans = deepcopy(a.route_plans)
    #new_vehicle_locations = deepcopy(s.vehicle_locations)
    new_route_plans = Vector{Vector{theta}}(undef, 0)
    new_vehicle_locations = Vector{theta}(undef, 0)
    
    # record actual_pickup_time and actual_dropoff_time
    new_requests = s.requests
    for v in 1:m.num_vehicles
        for th in a.route_plans[v]
            if th.p == 0
                new_requests[th.request_id].actual_pickup_time = th.a
            else
                new_requests[th.request_id].actual_dropoff_time = th.a
            end
            new_requests[th.request_id].vehicle_id = v
        end
    end
    
    # vehicles traverse the route plan
    for i in 1:m.num_vehicles
        visited = filter(x -> x.a <= new_time, a.route_plans[i])
        not_visited = filter(x -> x.a > new_time, a.route_plans[i])
        #println("vehicle: $(i)")
        #println("visited: $(visited)")
        #println("not_visited: $(not_visited)")
        
        # update route_plans and vehicle_locations
        new_vehicle_location = s.vehicle_locations[i]
        if length(visited) > 0
            new_vehicle_location = visited[length(visited)]
        end
        if (new_vehicle_location.a <= new_time) & (length(not_visited) > 0)
            new_vehicle_location = not_visited[1]
            popfirst!(not_visited)
        end
        push!(new_vehicle_locations, new_vehicle_location)
        push!(new_route_plans, not_visited)
    end

    sp = CentMDPState(new_time, new_requests, new_request_id, new_vehicle_locations, new_route_plans, false)
    return Deterministic(sp)
end
end

@everywhere begin
function POMDPs.reward(m::CentMDP, s::CentMDPState, a::CentAction, sp::CentMDPState)
    reward = 0.0
    for v in 1:m.num_vehicles
        reward = reward + length(a.route_plans[v]) - length(s.route_plans[v])
    end
    return reward
end
end

@everywhere begin
function GetInitialState(num_vehicles::Int64, test_chain_id::Int64, depot_id::Int64, test::Bool)
    requests = deepcopy(TEST_CHAINS[test_chain_id])
    request_id = 1
    time = requests[request_id].request_arrive_time
    
    vehicle_locations = Vector{theta}(undef, 0)
    route_plans = Vector{Vector{theta}}(undef, 0)
    for i in 1:num_vehicles
        push!(vehicle_locations, theta(time, -1, 10, 1, depot_id, 0, test_chain_id))
        push!(route_plans, [])
    end
    if test
        requests = requests[1:10]
    end
    return CentMDPState(time, requests, request_id, vehicle_locations, route_plans, false)
end

# simulation results
function GetSimMetrics(final_state::CentMDPState, total_reward::Float64, time_per_request::Float64, test_chain_id::Int64)
    total_travel_time = 0
    total_direct_travel_time = 0
    total_trips = 0
    total_trips_served = 0
    for request in final_state.requests
        if request.actual_dropoff_time != -1
            total_travel_time = total_travel_time + request.actual_dropoff_time - request.actual_pickup_time
            total_direct_travel_time = total_direct_travel_time + TRAVEL_TIME_MATRIX[request.pickup_node_id, request.dropoff_node_id]
            total_trips_served = total_trips_served + 1
        end
        #total_direct_travel_time = total_direct_travel_time + TRAVEL_TIME_MATRIX[request.pickup_node_id, request.dropoff_node_id]
        total_trips = total_trips + 1
    end
    service_rate = total_trips_served / total_trips
    TRT = total_travel_time
    TDT = total_direct_travel_time
    result = SimMetrics(service_rate, TRT, TDT, total_trips, total_trips_served, time_per_request, test_chain_id, 1)
    return result
end

struct SimMetrics
    service_rate::Float64
    TRT::Float64
    TDT::Float64
    total_trips::Int64
    total_trips_served::Int64
    time_per_request::Float64
    test_chain_id::Int64
    temp::Int64
end

POMDPs.isterminal(mdp::CentMDP, s::CentMDPState) = s.done
end

@everywhere begin
function RunSimulation(num_vehicles::Int64, 
        capacity::Int64, 
        k_max::Int64, 
        vv::Bool,
        budget::Bool,
        depot_id::Int64, 
        policy_type::String, 
        test_chain_id::Int64,
        mcts_iterations::Int64,
        mcts_depth::Int64,
        mcts_num_chains::Int64,
        mcts_runtime::Float64,
        test::Bool)
    
    mdp = CentMDP(num_vehicles, capacity, k_max, vv, budget)
    s = GetInitialState(num_vehicles, test_chain_id, depot_id, test)
    if policy_type == "random"
        solver = MyRandomSolver(100)
        policy = solve(solver, mdp)
    elseif policy_type == "greedy"
        solver = MyMyopicSolver(100)
        policy = solve(solver, mdp)
    elseif policy_type == "mcts"
        solver = RootMCTSSolver(mcts_iterations, mcts_depth, 2.0, mcts_num_chains, true, "fast", mcts_runtime)
        policy = solve(solver, mdp)
    elseif policy_type == "mctsbase"
        solver = MCTSSolver(n_iterations=mcts_iterations, depth=mcts_depth, exploration_constant=2.0, max_time=mcts_runtime, estimate_value=RolloutEstimator(MCTSRolloutSolver(100)))
        policy = solve(solver, mdp)
    end
    
    r_total = 0.0
    disc = 1.0
    sim_start_time = time()
    counter = 1
    while !isterminal(mdp, s)
        #println(s.route_plans)
        #println(".........")
        start_time = time()
        a = action(policy, s)
        sp = rand(transition(mdp, s, a))
        r = reward(mdp, s, a, sp)
        r_total += r
        s = sp
        #println("Done with request: $(counter), Reward: $(r), in $(time()-start_time) seconds")
        #flush(stdout)
        counter += 1
    end
    time_per_request = (time() - sim_start_time) / length(s.requests)
    metrics = GetSimMetrics(s, r_total, time_per_request, test_chain_id)
    #return s, metrics
    return metrics
end

function FullSim(num_vehicles::Int64, 
        capacity::Int64, 
        k_max::Int64, 
        vv::Bool,
        budget::Bool,
        depot_id::Int64, 
        policy_type::String, 
        parallel::Bool, 
        test::Bool, 
        file_path::String, 
        test_chain_ids::Vector{Int64}, 
        mcts_iterations::Int64, 
        mcts_depth::Int64, 
        mcts_num_chains::Int64, 
        mcts_runtime::Float64)
    results = [SimMetrics(1.0, 1.0, 1.0, 1, 1, 1.0, 1.0, 0) for x in 1:length(test_chain_ids)]
    if parallel
        #Threads.@threads for k = 1:length(test_chain_ids)
        #    test_chain_id = test_chain_ids[k]
        #    results[k] = RunSimulation(num_vehicles, capacity, k_max, vv, budget, depot_id, policy_type, test_chain_id, mcts_iterations, mcts_depth, mcts_num_chains, mcts_runtime, test)
        #end
        results = pmap(test_chain_id->RunSimulation(num_vehicles, capacity, k_max, vv, budget, depot_id, policy_type, test_chain_id, mcts_iterations, mcts_depth, mcts_num_chains, mcts_runtime, test), [x for x in 1:length(test_chain_ids)])
    else
        for k in 1:length(test_chain_ids)
            test_chain_id = test_chain_ids[k]
            println(test_chain_id)
            flush(stdout)
            start_time = time()
            results[k] = RunSimulation(num_vehicles, capacity, k_max, vv, budget, depot_id, policy_type, test_chain_id, mcts_iterations, mcts_depth, mcts_num_chains, mcts_runtime, test)
            
            println("FullSim: Done with simulation $(test_chain_id) in $(time()-start_time) seconds")
            flush(stdout)
            println("FullSim: service_rate: $(results[k].service_rate), TRT: $(results[k].TRT), TDT: $(results[k].TDT), total_trips: $(results[k].total_trips), total_trips_served: $(results[k].total_trips_served), time_per_request: $(results[k].time_per_request)")
            flush(stdout)
            println(".............")
            #flush(stdout)
            #println(results[k].temp)
            df = DataFrame(servicerate = map(x -> x.service_rate, results), TRT = map(x -> x.TRT, results), TDT = map(x -> x.TDT, results), totaltrips = map(x -> x.total_trips, results), totaltripsserved = map(x -> x.total_trips_served, results), timeperrequest = map(x -> x.time_per_request, results), testchainid = map(x -> x.test_chain_id, results))
            CSV.write(file_path, df)
        end
    end
    df = DataFrame(servicerate = map(x -> x.service_rate, results), TRT = map(x -> x.TRT, results), TDT = map(x -> x.TDT, results), totaltrips = map(x -> x.total_trips, results), totaltripsserved = map(x -> x.total_trips_served, results), timeperrequest = map(x -> x.time_per_request, results), testchainid = map(x -> x.test_chain_id, results))
    CSV.write(file_path, df)
    return results
end

POMDPs.discount(mdp::CentMDP) = 0.99
end


### MAIN ###
test_chain_ids = [x for x in 1:length(TEST_CHAINS)]
depot_id = 3607
test = false

dirpath_base = "data/results"

# Experiment 1
num_vehicles_list = [3, 4, 5]
capacity_list = [8]
k_max_list = [10]
vv_list = [true]
budget_list = [false, true]
policy_type_list = ["mcts", "greedy"]
mcts_iterations_list = [1000]
mcts_depth_list = [20]
mcts_runtime_list = [Inf, 30]

for num_vehicles in num_vehicles_list
    for capacity in capacity_list
        for k_max in k_max_list
            for vv in vv_list
                for budget in budget_list
                    for policy_type in policy_type_list
                        for mcts_iterations in mcts_iterations_list
                            for mcts_depth in mcts_depth_list
                                for mcts_runtime in mcts_runtime_list
                                    if policy_type == "mcts"
                                        mcts_num_chains = 25
                                        parallel = false
                                    else
                                        mcts_num_chains = 1
                                        parallel = true
                                    end
                                    file_path = "$(dirpath_base)/$(num_vehicles)_$(capacity)_$(k_max)_$(vv)_$(budget)_$(policy_type)_$(mcts_iterations)_$(mcts_depth)_$(mcts_num_chains)_$(mcts_runtime).csv"
                                    #file_path = "$(dirpath_base)/$(num_vehicles)_$(capacity)_$(k_max)_$(vv)_$(budget)_$(policy_type)_$(mcts_iterations)_$(mcts_depth)_$(mcts_num_chains)_$(mcts_runtime)_cong.csv"
                                    println(file_path)
                                    flush(stdout)
                                    results = FullSim(num_vehicles, capacity, k_max, vv, budget, depot_id, policy_type, parallel, test, file_path, test_chain_ids, mcts_iterations, mcts_depth, mcts_num_chains, mcts_runtime)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

println("done with all simulations")
flush(stdout)




