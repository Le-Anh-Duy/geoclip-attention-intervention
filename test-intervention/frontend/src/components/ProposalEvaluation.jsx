export default function ProposalEvaluation({ result, onSelect }) {
  if (!result) return null

  const hasGroundTruth = result.baseline.top1_distance_km != null
  const rows = [...result.evaluations].sort((left, right) => {
    if (hasGroundTruth) return (right.distance_improvement_km ?? -Infinity) - (left.distance_improvement_km ?? -Infinity)
    return right.proposal.objectness - left.proposal.objectness
  })

  return (
    <div className="results-panel proposal-evaluation">
      <div className="panel-title-row">
        <h2 className="panel-title">Independent proposal evaluation</h2>
        <span className="pill pill-blue">
          {result.unique_patch_mask_count}/{result.raw_proposal_count} unique masks
        </span>
      </div>
      <p className="hint proposal-evaluation-note">
        The baseline runs once. Each row is a separate intervention; proposal boxes are not unioned.
      </p>
      <div className="proposal-table-wrap">
        <table className="proposal-table">
          <thead>
            <tr>
              <th>Box</th>
              <th>Objectness</th>
              <th>Patches</th>
              <th>Intervention top-1</th>
              {hasGroundTruth && <th>Distance</th>}
              {hasGroundTruth && <th>Improvement</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ proposal, intervention, distance_improvement_km: improvement }) => {
              const top1 = intervention.predictions[0]
              return (
                <tr key={proposal.index} onClick={() => onSelect(proposal)} title="Click to select this box on the image">
                  <td>#{proposal.index + 1}</td>
                  <td>{proposal.objectness.toFixed(3)}</td>
                  <td>{proposal.patch_count}</td>
                  <td>{top1 ? `${top1.lat.toFixed(3)}, ${top1.lon.toFixed(3)}` : '—'}</td>
                  {hasGroundTruth && <td>{intervention.top1_distance_km?.toFixed(1)} km</td>}
                  {hasGroundTruth && (
                    <td className={improvement > 0 ? 'metric-positive' : improvement < 0 ? 'metric-negative' : ''}>
                      {improvement == null ? '—' : `${improvement > 0 ? '+' : ''}${improvement.toFixed(1)} km`}
                    </td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {!hasGroundTruth && <p className="hint">Enter ground truth to rank proposals by distance improvement.</p>}
    </div>
  )
}
