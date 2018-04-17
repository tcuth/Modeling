lc_final <- easy_read('learning_curve.final.psv')
lc_cv_final <- easy_read('cv_learning_curve.final.psv')

lc <- cbind(lc_cv_final,lc_final)
lc$iter <- 1:nrow(lc)
lc <- melt(lc, id.vars = 'iter', variable.name = 'data')

ggplot(lc, aes(iter,value)) + geom_line(aes(colour = data)) + ylim(c(-1,-0.2))

lc %>% group_by(data) %>% summarise(min=min(value))
order(lc[lc$data == "cv_test","value"])
