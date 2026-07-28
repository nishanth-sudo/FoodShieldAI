interface InspectionResultProps {
  inspection: {
    food_type: string;
    freshness_score: number;
    shelf_life_days: number;
    packaging_defects: any[];
    contamination_risks: Record<string, number>;
    xai_heatmap_url: string;
    report: string;
  };
}

export const InspectionResult = ({ inspection }: InspectionResultProps) => {
  return (
    <div>
      <h2>{inspection.food_type}</h2>
      <p>Freshness: {inspection.freshness_score}%</p>
      <p>Shelf Life: {inspection.shelf_life_days} days</p>
      <img src={inspection.xai_heatmap_url} alt="XAI Heatmap" />
      <div>{inspection.report}</div>
    </div>
  );
};
