# Anchor — Goal

## The goal

Anchor is a small helper that lives quietly on a Mac and holds a person to the thing they said they were going to do.

When you sit down to work, it asks out loud what you are working on. You answer in one ordinary spoken sentence —
"I need to finish the assignment tonight" — and then you forget it exists. There is no window to open, no list to
keep, no settings to fill in, and nothing to tick off. A small dot in the top bar is the only sign it is there.

While you are doing what you said, it says nothing at all. It is watching which window is in front of you, what that
window is called, and roughly what the screen looks like — and it is doing almost all of that thinking on your own
machine, for free. It understands that work is messy: if you said you were coding, then a search engine, a
question-and-answer site, or an AI chat are all part of coding for a while. It is looking at whether what you are
doing still *relates* to what you said, not at which app you happen to have open.

When you wander somewhere unrelated and stay there past a point — a point it works out from the kind of task you
described, not a fixed number — it does not pounce. It waits for a natural gap: you stopped typing, or the screen
went still, or you have been gone so long that waiting any longer would be silly. Then it speaks, once.

It opens with a joke at your expense. Not a mean one, and never about you as a person — always about the funny
distance between the sentence you said and the thing you are actually doing right now. Something like: "Twelve
minutes ago you were a person writing an assignment. You are now a man researching whether crabs can swim. Bold
pivot." Then, in the same breath, it makes you choose: is this a short break, or is this the new main thing?

If you try to slide past the question, it holds its ground exactly once — and this time there is no joke at all. It
simply repeats your own sentence back to you and asks if it is still true. One push, never two. Then it accepts
whatever you say, even if what you say is a bad idea. A helper that keeps arguing is a helper that gets deleted.

From there it goes one of three ways. If you say it is temporary, it takes the amount of time you asked for — twenty
minutes, half an hour, "till nine" — pauses everything, leaves you completely alone, and then comes back and pulls you
home with a plain, friendly line and no teasing, because it would be unfair to tease you for a break it gave you.
If you say this is genuinely the new main thing, it lets the old sentence go, takes your new one, and holds you to
that one exactly the same way. And if you say the old thing is already finished, it marks it done, congratulates you
in one line, and asks what is next — so a goal actually ends instead of hanging over you forever.

It speaks and listens in both English and Hindi, and it answers in whichever one you used. It never gets angry with
you and it never lets its jokes get sharper: if you sound irritated, it turns the humour down for the rest of the day
and does not turn it back up on its own. Some goals are not funny at all — anything that sounds serious, medical, or
money-related — and for those it simply says what it sees, plainly, with no joke. If it is not sure you have drifted,
it keeps quiet rather than accusing you.

Everything it knows stays on your laptop in one file. It never saves a picture of your screen anywhere. It never
sends your screen contents anywhere unless it genuinely cannot tell what is going on from the window name alone,
and even then it sends one small, shrunk picture and immediately forgets it. Nothing goes to any server that is not
OpenAI's, there is no account to make, and it costs a few paise an hour to run.

## What it must do

1. **Start invisible.** When it runs, it puts one small dot in the Mac's top bar and nothing else. No window opens,
   no icon appears at the bottom of the screen, and there is nothing to configure before it works.

2. **Ask for the goal out loud.** On starting a session it speaks a short question asking what you are working on,
   and listens for your answer. The very first time it ever runs, it also says once that its voice is
   computer-generated.

3. **Take one spoken sentence as the goal.** It listens to your answer in English or Hindi, writes down your exact
   words, and treats that sentence as the thing it will hold you to.

4. **Ask at most one follow-up question.** If your sentence is vague ("study"), it asks a single short question to
   pin it down, then stops asking. It never asks a second one.

5. **Work out how patient to be, from the task itself.** From your sentence it decides how long a wander is
   acceptable, what counts as a pause for this kind of work, and which places are a normal part of doing it. Coding
   gets more room than writing; watching a lecture gets the most, and for that it understands that not touching the
   keyboard means you are watching, not that you have stopped.

6. **Watch the whole desktop, not just the web browser.** Once a second it notices which program is in front, what
   its window is called, which website is open if it is a browser, how long since you last touched the keyboard or
   mouse, and roughly what the screen looks like.

7. **Do nothing when nothing has happened.** If the screen and the window are the same as a second ago, it thinks
   nothing, sends nothing, and costs nothing.

8. **Remember what it has already decided.** If you go back to a place it has already judged for this goal, it reuses
   its own earlier answer instead of asking again. After the first hour, most of your switching costs nothing at all.

9. **Judge by meaning, not by app name.** It decides whether what you are doing still relates to the sentence you
   said. If you said you were coding, then a search engine or an AI chat counts as coding for a while. If that same
   search engine leads you to a game, that does not count, even though it is the same program.

10. **Look at the screen only when it truly cannot tell.** If the window name is useless, or if the picture on screen
    changed a lot while the name stayed the same — which is how wandering inside one program hides — it takes one
    small, shrunk picture, uses it to decide, and then forgets it. It never saves a picture anywhere.

11. **Not react to a single moment.** Wandering has to add up before anything happens. Coming back to the task
    drains that build-up twice as fast as wandering fills it, so it forgives quickly.

12. **Wait for a natural gap before speaking.** It only talks when you have stopped typing for a moment, or the
    screen has gone still, or you have been away so long that waiting more would be pointless. It never speaks while
    you are on a call, while the Mac is in Do Not Disturb, or while the screen is locked.

13. **Open with a funny roast, then the question.** Its first line is a short joke — two sentences at most — about the
    gap between what you said and what you are doing, naming the actual thing and roughly how long. Then immediately:
    is this a short break, or the new main thing?

14. **Never make the joke about you.** The joke is always about the situation. It is never about your looks, your
    intelligence, your self-discipline as a personal failing, your relationships, your family, your money, or your
    weight, and it never swears.

15. **Turn the humour down if you do not like it, and never back up.** If your reply sounds irritated rather than
    amused, it becomes gentler for the rest of the day and does not become sharper again on its own.

16. **Skip the joke entirely when it would be wrong.** No joke when the goal sounds serious, medical, money-related
    or sad. No joke when it is not confident you have actually drifted — then it says something careful instead. No
    joke at all when it is muted.

17. **Make its jokes shorter each time.** The second time it interrupts you during the same goal, the joke is
    shorter. The third time it is barely a phrase. It never escalates.

18. **Hold its ground exactly once, with no joke.** If your answer dodges the question — including saying nothing at
    all — it repeats your own sentence back to you word for word in a plain, calm voice and asks if it is still true.
    It does this once and only once, and then it accepts whatever you say next.

19. **Handle "remind me later" properly.** If you ask for more time, it understands whatever way you say it —
    "twenty minutes", "half an hour", "till nine", or just "a bit" — and repeats the amount back so you know it
    landed. Then it stops watching completely and leaves you alone for exactly that long.

20. **Come back after the break, reliably and kindly.** When the time is up it speaks a plain line reminding you of
    your original sentence and asking if you are going back to it — with no joke. This must still work if you shut
    the lid, let the Mac sleep, or quit and reopen the app; if it comes back late it says so.

21. **Switch to a new goal when you say so.** If you say this is now the main thing, it lets the old sentence go,
    takes your new sentence as the goal, and watches that one the same way. It does not ask you a follow-up question
    in the middle of this.

22. **Finish a goal properly.** If you say the old thing is done, it marks it finished, says one short line of
    congratulation, and asks what is next.

23. **Speak and listen in English and Hindi.** It replies in whichever language you used, and its Hindi jokes are
    written in Hindi rather than translated from English ones.

24. **Never nag, no matter what.** It will not interrupt you more than four times in an hour whatever happens, and
    never more than once per interruption. It never interrupts during a break it agreed to.

25. **Let you shut it up and correct it.** The dot's menu has exactly three things: mute it, tell it that its last
    call was wrong, and quit. Telling it that it was wrong makes it less likely to make the same mistake again today.

26. **Remember across restarts, and stay on your machine.** Closing the lid or quitting and reopening picks up the
    same goal. Everything is kept in one file on your Mac. There is no account, no server of ours, and no syncing.
    Nothing about your screen is stored anywhere.

27. **Keep working when something is missing.** If the Mac has not given it permission to read window names, it uses
    just the program name and tells you which permission is missing instead of breaking. If the internet or the
    speaking voice fails, it falls back to the Mac's own built-in voice rather than going silent.

28. **Be quick, cheap and provably right.** Watching costs about half a paisa's worth of thinking per hour; a whole
    day of use costs a few rupees. It comes with its own tests, including tests that specifically check the jokes are
    never personal, never appear when the goal is a serious one, and get shorter on repeats.

## How we know it is finished

Anchor is finished when a fresh copy of the project, set up from scratch on a Mac with nothing but the accounts and
key listed in setup.md, runs with a single command and does every one of the twenty-eight things above from end to
end: it starts invisible, asks aloud, takes a spoken sentence in English or Hindi, stays quiet while you are on task,
notices a real wander without reacting to a single moment, waits for a genuine gap, opens with a joke that is about
the situation and never about the person, forces the choice, holds its ground exactly once without joking, and then
correctly runs all three branches — including a "remind me later" that survives the lid being closed and comes back
with a plain, friendly line. Every one of those behaviours is proven by automated tests that pass, including a
dedicated set of tests for joke safety and one that fakes a sleeping laptop to prove the delayed reminder still
fires. Nothing waits for a go-ahead: there is nothing here that is public, irreversible, or spends anyone else's
money, so the build runs straight through to the end.
