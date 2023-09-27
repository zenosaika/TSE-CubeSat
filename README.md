# TSE-CubeSat
- Project Structure
    - TLE เป็นโฟลเดอร์ไว้เก็บ TLE: Two Line Element ซึ่ง represent วงโคจรของดาวเทียม
    - polygon เป็นโฟลเดอร์ไว้เก็บ polygon ของ region ที่เราสนใจ (longitude, latitude)
    - tmp เป็นโฟลเดอร์ไว้เก็บโปรแกรมที่เขียนไว้แต่น่าจะไม่ได้ใช้แล้ว แต่ไม่อยากลบ
    - constellation_coverage.py คือโปรแกรมที่มี footprint animation & percent coverage heatmap
    - groundstation_dashboard.py คือโปรแกรมที่มี dashboard โชว์พวก positions & time slots for uplink/downlink

- Installation
`pip install requirements.txt`