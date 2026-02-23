ATLAS: Product Requirements Document
Product Vision
A voice-first AI secretary that manages hierarchical goals, tracks progress, learns personal patterns, and answers "what should I be doing right now?" Built for a high-performing individual with ADHD who needs decision automation, not task lists.
Not: An AGI buddy, life coach, or productivity guru
Is: A competent executive secretary that knows your goals, tracks your progress, and keeps you on track

Core Problem Statement
User Profile:

High-achieving software engineer with ADHD
Top 0.001% performer when in deep work
10+ competing interests (engineering, startups, fitness, creative pursuits)
Core challenges:

Decision paralysis from too many high-value options
High activation energy to start tasks (not distraction, just not starting)
Poor deadline prioritization despite knowing about them
Wants dynamic schedule that adapts to shifting priorities
Needs external structure that respects internal motivation states



Current Pain Points:

Spends mental energy deciding "what should I work on now?"
Hard time initiating tasks even when decision is made
Loses track of progress across multiple goal hierarchies
Life disruptions (trips, events) derail routines with no recovery system
Interests compete for time with no clear allocation strategy
Patterns exist but aren't visible or actionable


User Stories
Primary Workflows
As a user, I want to:

Ask what to do right now

"Hey Atlas, what am I supposed to be doing right now?"
Get immediate, confident answer based on schedule, priorities, energy
Answer should consider: time of day, energy level, deadlines, pattern history


Capture thoughts instantly via voice

"Hey Atlas, remind me to check deployment logs when I get to my desk"
"Remember this: the auth refactor needs token validation refactor first"
Zero friction capture while walking, cooking, anywhere
Retrieval via semantic search: "What did I say about auth last week?"


Check progress on goals

"How's my job search looking?"
"Am I on track for the Kafka deep-dive?"
"What's my gym streak?"
Get quantitative answers with context: "8/15 applications, need 7 more by month-end"


Dynamically shift priorities

"Atlas, I'm really into this system design topic right now. Prioritize learning over job apps this week."
System respects hyperfocus states when productive
Reverts to base priorities after specified time or on request


Break down overwhelming goals

"I need to apply to 15 companies but feel overwhelmed. Help me break this down."
Get hierarchical breakdown: Company research → Resume tailoring → Application submission
Tasks spread across multiple sessions to reduce activation energy


Track hierarchical goals

3 levels: Life goal → Milestone → Tasks
Example: "Senior engineer role" → "Pass tech screens" → "Solve 150 leetcode problems"
Each level has metrics, deadlines, progress tracking
Can ask about progress at any level


Weekly reflection

Automated retro every Sunday evening
Guided questions about energy, focus, satisfaction, what went well/poorly
System extracts patterns and suggests adjustments
"You avoided job apps until Friday again. Want me to schedule them earlier next week?"


See my day/week overview

"How's my day looking?"
"What's on the schedule this week?"
Get summary with time allocation by goal, meetings, blocked time


Set habits and track streaks

"Make gym 3x per week non-negotiable"
"Track my meditation streak"
System enforces minimums, celebrates streaks, flags breaks


Get pattern-based insights

System notices: "You're most productive Tuesday mornings"
System warns: "You slept 5 hours. Tomorrow will likely be low-energy."
System suggests: "You haven't done guitar in 2 weeks. Schedule this weekend?"




Core Features (v1 Priority)
1. Voice Interface ⭐ CRITICAL
Why: Zero-friction capture and query. User thinks out loud, system remembers.
Requirements:

Wake word: "Hey Atlas" (or similar custom trigger)
Natural language input via voice
Handles commands:

Capture: "Remind me to X", "Remember: Y"
Query: "What's next?", "How's X looking?", "Show my schedule"
Planning: "Break down X goal", "Prioritize Y this week"
Meta: "Why did you schedule X now?"


Text-to-speech responses
Works on Mac initially (laptop mic)
Future: Expandable to room-based Pis for home omnipresence

Success Criteria:

90%+ speech recognition accuracy
<2 second response time for simple queries
<5 second response time for complex queries
Can be used while walking, cooking, working (hands-free)


2. Hierarchical Goal System ⭐ CRITICAL
Why: User has multiple competing goals at different time scales. Flat to-do lists don't work.
Requirements:

3-level hierarchy:

Level 1: Life goals (6-12 month horizon)
Level 2: Milestones (1-3 month horizon)
Level 3: Concrete tasks/habits (daily/weekly)


Each level has:

Name, description
Deadline (optional)
Success metrics
Current progress
Parent/child relationships


Can query progress at any level:

"How's job search?" → Shows L1 + all children
"How's leetcode progress?" → Shows L3 metrics


Visualize as tree structure (future: voice + visual interface)

Example Structure:
L1: Get senior engineer role by June 2025
  ├─ L2: Pass technical screens (March 2025)
  │   ├─ L3: Solve 150 leetcode (80% medium success rate)
  │   └─ L3: Master 10 system designs
  ├─ L2: Build visible presence (April 2025)
  │   ├─ L3: Tweet 3x/week (500 followers)
  │   └─ L3: Write 2 technical blogs/month
  └─ L2: Apply to 30 companies (ongoing)
      └─ L3: Submit 5 applications/week
Success Criteria:

Can create/edit goals via voice or config file
Progress rolls up from L3 → L2 → L1
"How's X?" queries always return data, never "I don't know"
Alerts when L3 tasks aren't contributing to L2/L1 goals


3. Dynamic Priority Management ⭐ CRITICAL
Why: User's motivation and focus shift. System must adapt without losing long-term direction.
Requirements:

Base priority configuration (user-defined):

Each L1 goal has weight (% of time): "Job search: 40%, Learning: 30%..."
System allocates time based on weights


Temporary overrides:

"Prioritize X over Y this week" → System adjusts allocation
"I'm in hyperfocus on Z" → System backs off other demands
Overrides expire automatically or on request


Context-dependent value:

System asks WHY priority is shifting
Stores reason: "Excited about Kafka topic" vs "Anxious about interviews"
Uses context for future pattern learning



Success Criteria:

Can override priorities via voice
System confirms override and expiry: "Prioritizing learning 50% this week, back to 30% next Monday?"
Override history queryable: "When was I last this focused on learning?"
No override = follows base config (predictable default behavior)


4. Voice Note Capture & Semantic Search
Why: User has random thoughts/tasks while doing other things. Writing them down breaks flow.
Requirements:

Capture types:

Reminder: "Remind me to X at Y time/location/context"
Note: "Remember: [thought/insight/idea]"
Task: "Add to backlog: [task]"


Storage in vector database for semantic search
Retrieval via natural language:

"What did I say about the auth refactor?"
"When did I mention Jake last?"
"Find my notes on Kafka"


Returns relevant snippets with timestamps

Success Criteria:

Zero-friction capture (< 5 words to store a thought)
Search returns results even with fuzzy queries
Can search across all history, not just recent
Supports "I don't remember exact words but it was about X"


5. Pattern Learning & Insights
Why: User can't see their own patterns. System can detect and surface them.
Requirements:

Track patterns over time:

Energy: When is user high/low energy? (by hour, day of week)
Productivity: What conditions predict good/bad days?
Motivation cycles: How often does user naturally cycle through interests?
Triggers: What predicts avoidance vs execution?


Overfit to N=1 (this specific user):

"YOU are most productive Tuesday mornings"
"YOU avoid job apps when you have >4 meetings that week"
Not generalized, not crowd-sourced, just user's data


Proactive insights:

"You slept 5 hours. Based on 30 days of data, you have 80% chance of low productivity tomorrow."
"You haven't done guitar in 18 days. Your longest gap historically is 14 days before you feel off."
"You're on a 3-day coding streak. Historically you crash after day 4. Schedule lighter work Friday?"



Success Criteria:

At least 5 patterns learned per month
Patterns referenced in scheduling decisions
User can query: "What patterns have you noticed about X?"
Patterns update as user changes (not static)


6. Automated Weekly Retro
Why: Reflection is critical for ADHD but user won't do it unprompted. Automation solves this.
Requirements:

Triggers every Sunday evening (or user-configured time)
Guided question flow:

"How was your energy this week?" (1-5 scale)
"How was your focus?" (1-5)
"How satisfied are you with the week?" (1-5)
"What went well?"
"What didn't go well?"
"What blocked you from X?" (if goals missed)
"Any wins to celebrate?"


System analyzes responses:

Correlates with timeline data (what actually happened)
Detects patterns: "Low satisfaction + high focus = working on wrong things?"
Extracts action items: "You said you avoided job apps. Want me to schedule them earlier next week?"


Stores retro data for long-term pattern learning

Success Criteria:

Retro completes in < 5 minutes
User doesn't need to remember what happened (system shows them)
Generates at least 1 actionable insight per retro
Retro data used in next week's planning


7. Rule-Based Scheduling Adjustments
Why: Most rescheduling is deterministic logic, doesn't need LLM. Saves cost, faster execution.
Requirements:

Automatic rescheduling triggers:

Deadline urgency: Task due in <48 hours → Move to top priority
Energy mismatch: Low energy + high-energy task scheduled → Swap with low-energy task
Neglected goal: Goal not worked on in >N days → Inject into schedule
Calendar change: New meeting added → Reflow tasks around it
Pattern-based: "User is productive 9-11am" → Schedule hard tasks then


User can query: "Why did you move X?"
System explains: "You have a deadline Tuesday and you're low energy right now. I moved leetcode to tomorrow morning when you're sharp."

Success Criteria:

Rescheduling happens automatically (no user intervention)
Explanations are clear and data-driven
User can override: "No, I want to do X now" → System respects
No rescheduling loops (converges to stable schedule)


8. Progress Tracking & Metrics
Why: "How's it going?" needs quantitative answers, not vague feelings.
Requirements:

For each goal level, track:

Current value vs target value
Rate of progress (trajectory)
Time remaining to deadline
On-track status (red/yellow/green)


Queryable via voice:

"Am I on track for X?"
"How many more Y do I need to hit the goal?"
"Show me progress on all active goals"


Visual representation (future):

Progress bars
Burndown charts
Trend lines



Success Criteria:

Every goal has measurable progress
"How's X?" never returns "I don't know"
Alerts when falling behind: "You need 2 more job apps this week to stay on track"
Celebrates milestones: "You hit 100 leetcode problems! 50 to go."


9. Habit Tracking & Streaks
Why: User wants consistency on certain activities (gym, meditation). Streaks provide motivation.
Requirements:

Define habits with minimums:

"Gym: 3x per week minimum"
"Meditation: Daily"
"Guitar: 1x per week minimum"


Track streaks:

Current streak (days in a row)
Longest streak ever
Break alerts: "You're about to break your 14-day gym streak"


Enforcement levels:

Soft: "You usually do X by now. Want to schedule it?"
Medium: "You haven't done X in 4 days. This is unusual for you."
Hard: "X is non-negotiable. Blocking time now."


Queryable:

"What's my gym streak?"
"When did I last play guitar?"



Success Criteria:

Habits tracked automatically (user just does them, system logs)
Streaks visible and celebrated
Breaks flagged before they happen
User feels motivated by streaks, not guilted by breaks


10. Schedule Overview & Planning
Why: "What's my day/week look like?" is a fundamental secretary question.
Requirements:

Daily overview:

Time blocks by activity
Meetings from calendar
Goal allocation (% time on each goal)
Energy-appropriate task distribution


Weekly overview:

High-level time allocation
Upcoming deadlines
Neglected goals flagged


Can ask:

"How's my day looking?"
"What's on the schedule tomorrow?"
"Show me this week's plan"


Planning queries:

"Can I fit X into this week?"
"When's my next free 2-hour block?"



Success Criteria:

Overview available via voice (no need to open app/calendar)
Summary is concise (< 30 seconds to deliver)
Highlights what matters (deadlines, conflicts, opportunities)
Can drill down: "Tell me more about Tuesday afternoon"


Non-Functional Requirements
Performance

Voice response time: <2s for simple queries, <5s for complex
Database queries: <100ms
Speech-to-text: <1s (local Whisper)
System remains responsive even with months of data

Cost

Monthly operating cost: <$20
LLM API calls: Budget-controlled with hard limits
Graceful degradation if budget exceeded (fallback to templates)

Privacy

All data stored locally (Mac initially, self-hosted Pi later)
No cloud storage of personal data (calendar, notes, retros)
LLM API calls send minimal context (just query + relevant data)
User owns all data, can export/delete anytime

Reliability

System continues working if internet down (local inference)
No data loss on crashes (frequent commits to DB)
Backup system for all data (daily snapshots)
Graceful handling of API failures

Maintainability

Architecture is mutable (expect frequent changes)
Easy to add new patterns, rules, features
Config-driven behavior (user can tune without code changes)
Clear separation: voice → intent → logic → data

Extensibility

Voice interface can expand to multiple rooms (Pi Zeros)
Can add new integrations (Notion, GitHub, Slack) without redesign
Goal hierarchy can go deeper if needed (4+ levels)
Pattern learning can add new pattern types


Out of Scope (v1)
Explicitly NOT Building:

❌ AGI/sentient buddy with evolving personality
❌ Distraction monitoring/blocking (user said not needed)
❌ Automatic calendar blocking (use existing calendar)
❌ Task startup scripts (user handles their own setup)
❌ Daily plan generator (user wants more routine-based)
❌ Meeting transcription (defer to v2)
❌ Multi-room voice (defer to v2)
❌ Mobile app (defer to v2)
❌ Social accountability features
❌ Gamification beyond streaks

Maybe Later (v2+):

Visual dashboard (web/desktop UI)
Browser extension for research tracking
Code context engine (understand user's codebase)
Email/message drafting
Social media content assistant
Calendar integration (auto-accept, scheduling assistant)
File organization automation
Proactive notifications (beyond voice)


Success Metrics (How to Measure if ATLAS Works)
Week 4:

✅ Voice capture used 5+ times/day
✅ "What should I do now?" asked 3+ times/day
✅ At least 1 weekly retro completed
✅ User reports "less time deciding what to work on"

Month 3:

✅ Goal hierarchy reflects all active goals (not just some)
✅ Pattern learning shows at least 10 patterns
✅ User trusts schedule suggestions 80%+ of the time
✅ Time spent on "what to work on?" reduced by 50%+

Month 6:

✅ 10-15% measured productivity improvement (hours in flow state)
✅ All L1 goals have measurable progress
✅ User rates ATLAS as "genuinely useful, not a toy"
✅ System used daily without conscious effort (habitual)

Long-term (12+ months):

✅ User can't imagine working without ATLAS
✅ Visible progress on all major life goals
✅ Reduced anxiety from decision paralysis
✅ System feels like extension of user's brain


Open Questions & Design Decisions
To Be Decided:

Voice wake word: "Hey Atlas" vs "Atlas" vs custom
Retro cadence: Weekly vs bi-weekly? Sunday evening vs user-triggered?
Override expiry: Auto-revert after N days or require explicit confirmation?
Energy input: User self-reports or inferred from patterns?
Goal metric types: Numeric only or also boolean/qualitative?
Pattern confidence threshold: When to surface vs when to keep learning?
Habit enforcement: Soft suggestions vs hard blocking?
Integration priorities: Which external tools (Notion, GitHub) to support first?

User Will Decide During Build:

Exact voice interaction style (formal vs casual tone)
Retro question phrasing and depth
Override behavior and defaults
Pattern detection sensitivity
Habit enforcement levels


Technical Constraints (Recommendations, Not Requirements)
Recommended But Flexible:

Language: Go (fast, easy to deploy, good for services)
Database: SQLite + Qdrant (simple, local, no server needed)
Voice: Whisper (local STT) + system TTS or Piper
LLM: Claude API with budget controls
Deployment: Mac initially, Raspberry Pi for always-on later

Hard Constraints:

Must run locally (privacy requirement)
Must work offline for core features (resilience)
Must be cost-controlled (<$20/month)
Must be maintainable by one person


User Journey (Typical Day with ATLAS)
Morning (8:00 AM):
[Walking to bathroom]
User: "Hey Atlas, how's my day looking?"
ATLAS: "You have 2 meetings: standup at 10am, 1-on-1 with Jake at 2pm. 
        5 hours of productive time. Scheduled: 2 hours leetcode, 
        2 hours system design study, 1 hour job applications.
        
        Your energy is usually high Tuesday mornings. 
        I put hard tasks (leetcode) first."

User: "Sounds good."
Mid-Morning (10:30 AM):
[After standup]
User: "Hey Atlas, what's next?"
ATLAS: "Leetcode session, 2 hours blocked until 12:30.
        You're on problem 87 of 150. Want to start?"

User: "Yes."
[Opens laptop, starts working]
Afternoon (3:00 PM):
[Walking back from lunch]
User: "Hey Atlas, remind me to review Sarah's PR before end of day."
ATLAS: "Noted. Remind you about Sarah's PR before 5pm?"
User: "Yes."
Evening (6:00 PM):
[Cooking dinner]
User: "Hey Atlas, I'm really excited about this distributed systems topic.
      Can I spend more time on it this week?"
      
ATLAS: "Sure. Increasing learning to 50% this week, reducing job apps to 20%.
        That gives you 8 hours for distributed systems deep-dive.
        Back to normal next Monday?"
        
User: "Perfect."
Late Evening (8:00 PM):
ATLAS: "Hey Ashin, how was today? Quick check-in."
User: "Pretty good. Got a lot done on leetcode."

ATLAS: "Nice. Energy level today, 1 to 5?"
User: "4"

ATLAS: "Focus?"
User: "4"

ATLAS: "Logged. You're on a 3-day high-energy streak. 
        Based on your patterns, you usually crash on day 5.
        Want me to schedule lighter work Friday?"
        
User: "Yeah, good idea."

Summary for Cursor/Claude
Feed this entire document to your LLM.
Key points to emphasize:

This is a voice-first secretary system, not an AGI
Core user problem: decision paralysis + task initiation friction
Must be mutable architecture (user will iterate heavily)
Hierarchical goals (3 levels) are fundamental
Pattern learning should overfit to this one user (N=1)
Voice is critical - zero friction capture/query
Cost-controlled (<$20/month hard limit)
Runs locally (privacy + resilience)
Success = 10-15% productivity improvement, not AGI magic
User will refine requirements as they build/use it

Start with: Voice interface + hierarchical goals + basic query answering
Add next: Pattern learning + weekly retros
Polish later: Advanced scheduling, integrations, multi-room voice
This is a build-to-think project - requirements will evolve as user discovers what actually helps vs what sounded good on paper.