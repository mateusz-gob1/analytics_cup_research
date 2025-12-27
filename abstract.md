# Research Track Abstract

## Introduction
A simple metric is presented to measure how well football players move together using tracking data. Tracking data provide player positions over time, but describing "team movement" with a single number is not straightforward. The goal is to offer an easy and interpretable measure of coordination that analysts can compute quickly. The method works both for small local groups (triangles) and for any user-chosen group of players, using only open SkillCorner data. The metric is called the Group Synchronization Index (GSI).

## Methods
For each player, speed, movement direction, and changes in acceleration (impulse) are computed. Directional alignment is measured by comparing player directions relative to the ball (inspired by Kuramoro synchronization model), which produces a score between 0 and 1 (1 means everyone moves in the same direction). This ball-relative view reduces bias from absolute pitch orientation and focuses on collective intent. Three parts are combined into one final synchrony score: direction alignment, speed consistency, and impulse consistency. Speed consistency checks whether players move at a similar pace, while impulse consistency checks whether players start or adjust movement at the same time. The final score is a weighted blend of these terms and is normalized to the 0 to 1 range. Default weights are provided but can be adjusted for different analysis goals.

The score can be computed for Delaunay triangles or for any custom list of player IDs. The toolkit includes Delaunay triangle extraction, time-window selection, and pitch visualizations, both as static plots and animations. This makes it possible to move from a numeric score to a visual explanation of what is happening on the pitch.

## Results
The proposed metric produces smooth and interpretable synchrony signals over time for both local player groups and larger team units. Across different group definitions, GSI highlights moments of high and low coordination and helps identify frames in which collective movement is most pronounced. High synchronization values typically appear when players adjust their movement direction and timing together in relation to the ball, while low values indicate more independent or uncoordinated behavior. Visualizations and animations help explain these patterns by linking the numeric score to player positions on the pitch.

## Conclusion
The proposed metric and tools provide a clear and practical way to study coordinated movement in football. The method is transparent, configurable, and works for both local triangles and larger groups. Because it emphasizes timing and alignment rather than raw speed, it is less sensitive to individual physical profiles. This makes it useful for analysts who want to explore team behavior without heavy custom preprocessing. The toolkit also provides a solid starting point for future studies of pressing, collective shifts, and positional organization.
