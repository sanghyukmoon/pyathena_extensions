# On-the-fly analysis project coordination

For core_formation work in this project, read /home/sm69/athena_radial_profile/AGENTS.md, /home/sm69/athena_radial_profile/PLANS.md, and /home/sm69/athena_radial_profile/plans/onthefly-analysis-master.md before planning or implementation. Also follow core_formation/AGENTS.md within that package.

The single shared output-time specification is /home/sm69/athena_radial_profile/OUTPUT_CONTRACT.md. Read it when changing timelines, loaders, numbering, or mixed-data consumers. Do not duplicate it here or assume target behavior is already implemented. Contract revisions require reviewing Athena++, base readers, application consumers, tests, and old-output compatibility together.

Every future project planning session must update the master plan and its dedicated smaller plan; every implementation milestone must update both. Use the master's terminology: t_coll is sink-particle formation, t_crit is the onset of runaway collapse, r_crit/M_crit are virial-based quantities, and r_TES/M_TES are TES predictions. Scientific formulas and existing identifiers are not to be changed merely for terminology. Unresolved projection physics requires user review.

Preserve unrelated user edits. Use the pyathena Python environment and place all new simulation/analysis artifacts only under /scratch/gpfs/sm69/onthefly-rprof-test. Keep focused commits with Co-authored-by: Codex <codex@openai.com>. Update explanatory PDFs after major milestones.
