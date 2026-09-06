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
        <h2 className="panel-title">Đánh giá từng proposal độc lập</h2>
        <span className="pill pill-blue">
          {result.unique_patch_mask_count}/{result.raw_proposal_count} mask duy nhất
        </span>
      </div>
      <p className="hint proposal-evaluation-note">
        Baseline chỉ chạy một lần. Mỗi hàng bên dưới là một lần intervention riêng, không union các box.
      </p>
      <div className="proposal-table-wrap">
        <table className="proposal-table">
          <thead>
            <tr>
              <th>Box</th>
              <th>Objectness</th>
              <th>Patch</th>
              <th>Top-1 sau intervention</th>
              {hasGroundTruth && <th>Khoảng cách</th>}
              {hasGroundTruth && <th>Cải thiện</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ proposal, intervention, distance_improvement_km: improvement }) => {
              const top1 = intervention.predictions[0]
              return (
                <tr key={proposal.index} onClick={() => onSelect(proposal)} title="Bấm để chọn box này trên ảnh">
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
      {!hasGroundTruth && <p className="hint">Nhập ground truth để xếp hạng proposal theo mức cải thiện khoảng cách.</p>}
    </div>
  )
}
